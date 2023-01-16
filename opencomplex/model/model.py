# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
import weakref

import torch
import torch.nn as nn

from opencomplex.data.data_transforms import (
    pseudo_beta_fn,
)
from opencomplex.model.embedders import (
    InputEmbedder,
    RecyclingEmbedder,
    TemplateAngleEmbedder,
    TemplatePairEmbedder,
    ExtraMSAEmbedder,
    HMMEmbedder,
    SecondaryStructureEmbedder,
)
from opencomplex.model.evoformer import (
    EvoformerStack,
    ExtraMSAStack,
    PairTransformerStack,
)
from opencomplex.model.heads import (
    AuxiliaryHeads,
    AuxiliaryHeadsRNA,
)
from opencomplex.model.structure_module import StructureModule
from opencomplex.model.structure_module_rna import StructureModuleRNA
from opencomplex.model.structure_module_xyz import StructureModuleXYZ
from opencomplex.model.template import (
    TemplatePairStack,
    TemplatePointwiseAttention,
    embed_templates,
)
import opencomplex.np.nucleotide_constants as nucleotide_constants
import opencomplex.np.residue_constants as residue_constants
from opencomplex.utils.feats import (
    build_extra_msa_feat,
    dense_atom_to_all_atom,
)
from opencomplex.utils.tensor_utils import (
    add,
    dict_multimap,
    tensor_tree_map,
)
from opencomplex.utils.complex_utils import ComplexType, split_protein_rna_pos


class OpenComplex(nn.Module):
    """
    Alphafold 2.

    Implements Algorithm 2 (but with training).
    """

    def __init__(self, config, complex_type):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super(OpenComplex, self).__init__()

        self.complex_type = complex_type

        self.globals = config.globals
        self.config = config.model
        self.template_config = self.config.template
        self.extra_msa_config = self.config.extra_msa

        # Main trunk + structure module
        self.input_embedder = InputEmbedder(
            **self.config["input_embedder"],
        )
        self.recycling_embedder = RecyclingEmbedder(
            **self.config["recycling_embedder"],
        )

        self.hmm_embedder_enabled = self.ss_embedder_enabled = False
        if "hmm_embedder" in self.config:
            self.hmm_embedder_enabled = True
            self.hmm_embedder = HMMEmbedder(
                **self.config["hmm_embedder"],
            )
        if "ss_embedder" in self.config:
            self.ss_embedder_enabled = True
            self.ss_embedder = SecondaryStructureEmbedder(
                **self.config["ss_embedder"],
            )
            self.pair_transformer_stack = PairTransformerStack(
                **self.config["pair_transformer_stack"],
            )
        
        if(self.template_config.enabled):
            self.template_angle_embedder = TemplateAngleEmbedder(
                **self.template_config["template_angle_embedder"],
            )
            self.template_pair_embedder = TemplatePairEmbedder(
                **self.template_config["template_pair_embedder"],
            )
            self.template_pair_stack = TemplatePairStack(
                **self.template_config["template_pair_stack"],
            )
            self.template_pointwise_att = TemplatePointwiseAttention(
                **self.template_config["template_pointwise_attention"],
            )
       
        if(self.extra_msa_config.enabled):
            self.extra_msa_embedder = ExtraMSAEmbedder(
                **self.extra_msa_config["extra_msa_embedder"],
            )
            self.extra_msa_stack = ExtraMSAStack(
                **self.extra_msa_config["extra_msa_stack"],
            )
        
        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
        )

        if self.complex_type == ComplexType.PROTEIN:
            self.structure_module = StructureModule(
                **self.config["structure_module"],
            )
            self.aux_heads = AuxiliaryHeads(
                self.config["heads"],
            )
        elif self.complex_type == ComplexType.RNA:
            self.structure_module = StructureModuleRNA(
                **self.config["structure_module"],
            )
            self.aux_heads = AuxiliaryHeadsRNA(
                self.config["heads"],
            )
        else:
            self.structure_module = StructureModuleXYZ(
                **self.config["structure_module"],
            )
            self.aux_heads = AuxiliaryHeadsRNA(
                self.config["heads"],
            )
            

    def iteration(self, feats, prevs, _recycle=True):
        # Primary output dictionary
        outputs = {}

        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in feats:
            if(feats[k].dtype == torch.float32):
                feats[k] = feats[k].to(dtype=dtype)

        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        no_batch_dims = len(batch_dims)
        n = feats["target_feat"].shape[-2]
        n_seq = feats["msa_feat"].shape[-3]
        device = feats["target_feat"].device
        
        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        # Prep some features
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]
        
        asym_id = feats.get("asym_id", None)
        sym_id = feats.get("sym_id", None)
        entity_id = feats.get("entity_id", None)

        ## Initialize the MSA and pair representations

        # m: [*, S_c, N, C_m]
        # z: [*, N, N, C_z]
        m, z = self.input_embedder(
            feats["target_feat"],
            feats["residue_index"],
            feats["msa_feat"],
            inplace_safe=inplace_safe,
            asym_id=asym_id,
            sym_id=sym_id,
            entity_id=entity_id,
        )

        if self.hmm_embedder_enabled:
            hmm_embed = self.hmm_embedder(feats["hmm"])
            # the dimension of m is [*, N_seq + 1, N_res, C_m] due to the add hmm embedding
            m = torch.cat(
                [
                    m,
                    hmm_embed[..., None, :, :]
                ],
                dim=-3
            )
            # the dimension of msa_mask is [*, N_seq + 1, N_res] due to the add hmm embedding
            msa_mask = torch.cat(
                [
                    msa_mask,
                    msa_mask.new_ones((*batch_dims, 1, n)),
                ],
                dim=-2
            )

            del hmm_embed

        if self.ss_embedder_enabled:
            ss_embed = self.ss_embedder(feats["ss"])
            ss_embed = self.pair_transformer_stack(
                ss_embed,
                chunk_size=self.globals.chunk_size,
                use_lma=self.globals.use_lma,
                pair_mask=pair_mask.to(dtype=m.dtype),
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans,
            )
            z = add(z, ss_embed, inplace=inplace_safe)

            del ss_embed

        # Unpack the recycling embeddings. Removing them from the list allows 
        # them to be freed further down in this function, saving memory
        m_1_prev, z_prev, x_prev = reversed([prevs.pop() for _ in range(3)])

        # Initialize the recycling embeddings, if needs be 
        if None in [m_1_prev, z_prev, x_prev]:
            # [*, N, C_m]
            m_1_prev = m.new_zeros(
                (*batch_dims, n, self.config.input_embedder.c_m),
                requires_grad=False,
            )

            # [*, N, N, C_z]
            z_prev = z.new_zeros(
                (*batch_dims, n, n, self.config.input_embedder.c_z),
                requires_grad=False,
            )

            # [*, N, 3]
            x_prev = z.new_zeros(
                (*batch_dims, n, residue_constants.atom_type_num, 3),
                requires_grad=False,
            )

        x_prev = pseudo_beta_fn(
            feats, feats["butype"], x_prev, None, complex_type=self.complex_type
        ).to(dtype=z.dtype)

        # The recycling embedder is memory-intensive, so we offload first
        if(self.globals.offload_inference and inplace_safe):
            m = m.cpu()
            z = z.cpu()

        # m_1_prev_emb: [*, N, C_m]
        # z_prev_emb: [*, N, N, C_z]
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
            x_prev,
            inplace_safe=inplace_safe,
        )

        if(self.globals.offload_inference and inplace_safe):
            m = m.to(m_1_prev_emb.device)
            z = z.to(z_prev.device)

        # [*, S_c, N, C_m]
        m[..., 0, :, :] += m_1_prev_emb

        # [*, N, N, C_z]
        z = add(z, z_prev_emb, inplace=inplace_safe)

        # Deletions like these become significant for inference with large N,
        # where they free unused tensors and remove references to others such
        # that they can be offloaded later
        del m_1_prev, z_prev, x_prev, m_1_prev_emb, z_prev_emb

        # Embed the templates + merge with MSA/pair embeddings
        if self.config.template.enabled: 
            if asym_id is not None:
                multichain_mask = asym_id[..., None] == asym_id[..., None, :]
            else:
                multichain_mask = None

            if "template_mask" not in feats.keys():
                feats["template_mask"] = torch.ones(
                    feats["template_butype"].shape[:no_batch_dims+1], dtype=m.dtype, device=device
                )

            template_feats = {
                k: v for k, v in feats.items() if k.startswith("template_")
            }
            template_embeds = embed_templates(
                self,
                template_feats,
                z,
                pair_mask.to(dtype=z.dtype),
                no_batch_dims,
                inplace_safe=inplace_safe,
                multichain_mask=multichain_mask,
            )

            # [*, N, N, C_z]
            z = add(z,
                template_embeds.pop("template_pair_embedding"),
                inplace_safe,
            )

            if "template_angle_embedding" in template_embeds:
                # [*, S = S_c + S_t, N, C_m]
                m = torch.cat(
                    [m, template_embeds["template_angle_embedding"]], 
                    dim=-3
                )

                # [*, S, N]
                torsion_angles_mask = feats["template_torsion_angles_mask"]
                msa_mask = torch.cat(
                    [feats["msa_mask"], torsion_angles_mask[..., 2]], 
                    dim=-2
                )

        # Embed extra MSA features + merge with pairwise embeddings
        if self.config.extra_msa.enabled:
            # [*, S_e, N, C_e]
            a = self.extra_msa_embedder(build_extra_msa_feat(feats, self.complex_type))

            if(self.globals.offload_inference):
                # To allow the extra MSA stack (and later the evoformer) to
                # offload its inputs, we remove all references to them here
                input_tensors = [a, z]
                del a, z
    
                # [*, N, N, C_z]
                z = self.extra_msa_stack._forward_offload(
                    input_tensors,
                    msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                    chunk_size=self.globals.chunk_size,
                    use_lma=self.globals.use_lma,
                    pair_mask=pair_mask.to(dtype=m.dtype),
                    _mask_trans=self.config._mask_trans,
                )
    
                del input_tensors
            else:
                # [*, N, N, C_z]
                z = self.extra_msa_stack(
                    a, z,
                    msa_mask=feats["extra_msa_mask"].to(dtype=m.dtype),
                    chunk_size=self.globals.chunk_size,
                    use_lma=self.globals.use_lma,
                    pair_mask=pair_mask.to(dtype=m.dtype),
                    inplace_safe=inplace_safe,
                    _mask_trans=self.config._mask_trans,
                )

        # Run MSA + pair embeddings through the trunk of the network
        # m: [*, S, N, C_m]
        # z: [*, N, N, C_z]
        # s: [*, N, C_s]          
        if(self.globals.offload_inference):
            input_tensors = [m, z]
            del m, z
            m, z, s = self.evoformer._forward_offload(
                input_tensors,
                msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
                pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
                chunk_size=self.globals.chunk_size,
                use_lma=self.globals.use_lma,
                _mask_trans=self.config._mask_trans,
            )
    
            del input_tensors
        else:
            m, z, s = self.evoformer(
                m,
                z,
                msa_mask=msa_mask.to(dtype=m.dtype),
                pair_mask=pair_mask.to(dtype=z.dtype),
                chunk_size=self.globals.chunk_size,
                use_lma=self.globals.use_lma,
                use_flash=self.globals.use_flash,
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans,
            )

        outputs["msa"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s

        del z

        poses = {}

        if self.complex_type == ComplexType.MIX:
            protein_pos, rna_pos = split_protein_rna_pos(feats, self.complex_type)
            poses = {"protein_pos": protein_pos, "rna_pos": rna_pos}

        # Predict 3D structure
        outputs["sm"] = self.structure_module(
            outputs,
            feats["butype"],
            mask=feats["seq_mask"].to(dtype=s.dtype),
            **poses,
            inplace_safe=inplace_safe,
            _offload_inference=self.globals.offload_inference,
        )

        outputs["final_atom_positions"] = dense_atom_to_all_atom(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["all_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        # Save embeddings for use during the next recycling iteration

        # [*, N, C_m]
        m_1_prev = m[..., 0, :, :]

        # [*, N, N, C_z]
        z_prev = outputs["pair"]

        # [*, N, 3]
        x_prev = outputs["final_atom_positions"]

        (
            outputs["pred_pseudo_beta"],
            outputs["pred_pseudo_beta_mask"]
        ) = pseudo_beta_fn(
            feats,
            feats["butype"],
            outputs["final_atom_positions"],
            outputs["final_atom_mask"],
            self.complex_type
        )

        return outputs, m_1_prev, z_prev, x_prev

    def forward(self, batch):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "butype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "residue_index" ([*, N_res])
                        Tensor whose final dimension consists of
                        consecutive indices from 0 to N_res.
                    "msa_feat" ([*, N_seq, N_res, C_msa])
                        MSA features, constructed as in the supplement.
                        C_msa is config.model.input_embedder.msa_dim.
                    "seq_mask" ([*, N_res])
                        1-D sequence mask
                    "msa_mask" ([*, N_seq, N_res])
                        MSA mask
                    "pair_mask" ([*, N_res, N_res])
                        2-D pair mask
                    "extra_msa_mask" ([*, N_extra, N_res])
                        Extra MSA mask
                    "template_mask" ([*, N_templ])
                        Template mask (on the level of templates, not
                        residues)
                    "template_butype" ([*, N_templ, N_res])
                        Tensor of template residue indices (indices greater
                        than 19 are clamped to 20 (Unknown))
                    "template_all_atom_positions"
                        ([*, N_templ, N_res, 37, 3])
                        Template atom coordinates in atom37 format
                    "template_all_atom_mask" ([*, N_templ, N_res, 37])
                        Template atom coordinate mask
                    "template_pseudo_beta" ([*, N_templ, N_res, 3])
                        Positions of template carbon "pseudo-beta" atoms
                        (i.e. C_beta for all residues but glycine, for
                        for which C_alpha is used instead)
                    "template_pseudo_beta_mask" ([*, N_templ, N_res])
                        Pseudo-beta mask
        """
        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev = None, None, None
        prevs = [m_1_prev, z_prev, x_prev]

        is_grad_enabled = torch.is_grad_enabled()

        # Main recycling loop
        num_iters = batch["butype"].shape[-1]
        for cycle_no in range(num_iters): 
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batch)

            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # Run the next iteration of the model
                outputs, m_1_prev, z_prev, x_prev = self.iteration(
                    feats,
                    prevs,
                    _recycle=(num_iters > 1)
                )

                if(not is_final_iter):
                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev]
                    del m_1_prev, z_prev, x_prev

        # Run auxiliary heads
        outputs.update(self.aux_heads(outputs))

        return outputs
