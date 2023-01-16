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
import torch
import torch.nn as nn

from opencomplex.utils.affine_utils import AffineTransformation    
from opencomplex.utils.feats_rna import (
    frames_and_literature_positions_to_atom23_pos,
    torsion_angles_to_frames,
)

from opencomplex.model.primitives import Linear, LayerNorm

from opencomplex.np.nucleotide_constants import (
    nttype_rigid_group_default_frame,
    nttype_atom23_to_rigid_group,
    nttype_atom23_mask,
    nttype_atom23_rigid_group_positions
)
from opencomplex.model.structure_module import (
    AngleResnet,
    StructureModuleTransition,
    InvariantPointAttention,
)
from opencomplex.utils.tensor_utils import (
    dict_multimap,
)

class StructureModuleRNA(nn.Module):
    def __init__(
        self,
        c_s,
        c_z,
        c_ipa,
        c_resnet,
        no_heads_ipa,
        no_qk_points,
        no_v_points,
        dropout_rate,
        no_blocks,
        no_transition_layers,
        no_resnet_blocks,
        no_angles,
        trans_scale_factor,
        epsilon,
        inf,
        **kwargs,
    ):
        """
        Args:
            c_s:
                Single representation channel dimension
            c_z:
                Pair representation channel dimension
            c_ipa:
                IPA hidden channel dimension
            c_resnet:
                Angle resnet (Alg. 23 lines 11-14) hidden channel dimension
            no_heads_ipa:
                Number of IPA heads
            no_qk_points:
                Number of query/key points to generate during IPA
            no_v_points:
                Number of value points to generate during IPA
            dropout_rate:
                Dropout rate used throughout the layer
            no_blocks:
                Number of structure module blocks
            no_transition_layers:
                Number of layers in the single representation transition
                (Alg. 23 lines 8-9)
            no_resnet_blocks:
                Number of blocks in the angle resnet
            no_angles:
                Number of angles to generate in the angle resnet
            trans_scale_factor:
                Scale of single representation transition hidden dimension
            epsilon:
                Small number used in angle resnet normalization
            inf:
                Large number used for attention masking
        """
        super(StructureModuleRNA, self).__init__()

        self.c_s = c_s
        self.c_z = c_z
        self.c_ipa = c_ipa
        self.c_resnet = c_resnet
        self.no_heads_ipa = no_heads_ipa
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.dropout_rate = dropout_rate
        self.no_blocks = no_blocks
        self.no_transition_layers = no_transition_layers
        self.no_resnet_blocks = no_resnet_blocks
        self.no_angles = no_angles
        self.trans_scale_factor = trans_scale_factor
        self.epsilon = epsilon
        self.inf = inf

        # To be lazily initialized later
        self.default_frames = None
        self.group_idx = None
        self.atom_mask = None
        self.lit_positions = None

        self.layer_norm_s = LayerNorm(self.c_s)
        self.layer_norm_z = LayerNorm(self.c_z)

        self.linear_in = Linear(self.c_s, self.c_s)

        self.ipa = InvariantPointAttention(
            self.c_s,
            self.c_z,
            self.c_ipa,
            self.no_heads_ipa,
            self.no_qk_points,
            self.no_v_points,
            inf=self.inf,
            eps=self.epsilon,
        )

        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)

        self.transition = StructureModuleTransition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate,
        )

        # refine module
        self.refine_net = Linear(self.c_s, 5*3, init="final")

        # sidechain
        self.angle_resnet = AngleResnet(self.c_s,
            self.c_resnet,
            self.no_resnet_blocks,
            self.no_angles,
            self.epsilon,
        )

    def forward(
        self,
        evoformer_output_dict,
        rttype,
        mask=None,
        inplace_safe=False,
        _offload_inference=False,
    ):
        """
        Args:
            evoformer_output_dict:
                Dictionary containing:
                    "single":
                        [*, N_res, C_s] single representation
                    "pair":
                        [*, N_res, N_res, C_z] pair representation
            rttype:
                [*, N_res] amino acid indices
            mask:
                Optional [*, N_res] sequence mask
        Returns:
            A dictionary of outputs
        """
        s = evoformer_output_dict["single"]

        if mask is None:
            # [*, N]
            mask = s.new_ones(s.shape[:-1])
        
        # [*, N, C_s]
        B, N = s.shape[:2]
        s = self.layer_norm_s(s)

        # [*, N, N, C_z]
        z = self.layer_norm_z(evoformer_output_dict["pair"])

        z_reference_list = None
        if(_offload_inference):
            evoformer_output_dict["pair"] = evoformer_output_dict["pair"].cpu()
            z_reference_list = [z]
            z = None

        # [*, N, C_s]
        s_initial = s
        s = self.linear_in(s)

        # black hole initialization
        xyz = torch.zeros((B, N, 5, 3)).to(s.device)
        
        outputs = []
        frames = []
        sidechain_frames = []
        for i in range(self.no_blocks):
            # O4' = xyz[:,:,0,:]
            # C4' = xyz[:,:,1,:]
            # C3' = xyz[:,:,2,:]
            # C1' = xyz[:,:,3,:]
            # C2' = xyz[:,:,4,:]

            s = s + self.ipa(
                s, 
                z, 
                mask, 
                xyz=xyz,
                inplace_safe=inplace_safe,
                _offload_inference=_offload_inference, 
                _z_reference_list=z_reference_list
            )
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

            # backbone refine
            xyz_update = self.refine_net(s)
            xyz_update = xyz_update.view(B, N, 5, 3)
            xyz = xyz + xyz_update

            # predict sidechain
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            # all frames
            bb1_to_global = AffineTransformation.from_3_points_pos(xyz[:,:,0,:], xyz[:,:,1,:], xyz[:,:,2,:])
            bb2_to_global = AffineTransformation.from_3_points_pos(xyz[:,:,0,:], xyz[:,:,3,:], xyz[:,:,4,:])
            bb_to_global = AffineTransformation.cat([bb1_to_global.unsqueeze(-1), bb2_to_global.unsqueeze(-1)], dim=-1)
            bb_to_global.t = bb_to_global.t*self.trans_scale_factor
            
            all_frames_to_global = self.torsion_angles_to_frames(
                bb_to_global, 
                angles,
                rttype
            )
            
            # result
            pred_xyz = self.frames_and_literature_positions_to_atom23_pos(
                all_frames_to_global,
                rttype.to(torch.long),
            )

            preds = {
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
            }

            outputs.append(preds)
            frames.append(bb_to_global.unsqueeze(0))
            sidechain_frames.append(all_frames_to_global.unsqueeze(0))
        del z, z_reference_list
        
        if(_offload_inference):
            evoformer_output_dict["pair"] = (
                evoformer_output_dict["pair"].to(s.device)
            )

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s
        
        outputs['frames'] = AffineTransformation.cat(frames, dim=0)
        outputs['sidechain_frames'] = AffineTransformation.cat(sidechain_frames, dim=0)

        return outputs

    def _init_residue_constants(self, float_dtype, device):
        if self.default_frames is None:
            self.default_frames = torch.tensor(
                nttype_rigid_group_default_frame,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.group_idx is None:
            self.group_idx = torch.tensor(
                nttype_atom23_to_rigid_group,
                device=device,
                requires_grad=False,
            )
        if self.atom_mask is None:
            self.atom_mask = torch.tensor(
                nttype_atom23_mask,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
        if self.lit_positions is None:
            self.lit_positions = torch.tensor(
                nttype_atom23_rigid_group_positions,
                dtype=float_dtype,
                device=device,
                requires_grad=False,
            )
            
    def torsion_angles_to_frames(self, frames, alpha, nts):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(frames, alpha, nts, self.default_frames)
            
    def frames_and_literature_positions_to_atom23_pos(
        self, frames, nts
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(frames.r.dtype, frames.r.device)
        return frames_and_literature_positions_to_atom23_pos(
            frames,
            nts,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
