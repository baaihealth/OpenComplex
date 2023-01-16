# Copyright 2022 BAAI
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

from opencomplex.utils.feats_rna import (
    frames_and_literature_positions_to_atom23_pos,
    torsion_angles_to_frames,
)

from opencomplex.utils.rigid_utils import Rigid

from opencomplex.np.nucleotide_constants import (
    nttype_rigid_group_default_frame,
    nttype_atom23_to_rigid_group,
    nttype_atom23_mask,
    nttype_atom23_rigid_group_positions
)
from opencomplex.model.sm.structure_module import StructureModule, BackboneUpdate

from opencomplex.utils.tensor_utils import (
    dict_multimap,
)

class StructureModuleRNA(StructureModule):
    def __init__(self, *args, **kwargs):
        super(StructureModuleRNA, self).__init__(*args, **kwargs)
        self.bb_update = BackboneUpdate(self.c_s, 5*3)

    def forward(
        self,
        evoformer_output_dict,
        rttype,
        mask=None,
        inplace_safe=False,
        _offload_inference=False,
        **kwargs,
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
        xyz = torch.zeros((s.shape[:-1] + (5, 3))).to(s.device)
        
        outputs = []
        frames = []
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
            xyz_update = self.bb_update(s)
            xyz_update = xyz_update.view(xyz.shape)
            xyz = xyz + xyz_update

            # predict sidechain
            unnormalized_angles, angles = self.angle_resnet(s, s_initial)

            # all frames
            bb1_to_global = Rigid.from_3_points(-xyz[...,0,:], xyz[...,1,:], xyz[...,2,:])
            bb2_to_global = Rigid.from_3_points(-xyz[...,0,:], xyz[...,3,:], xyz[...,4,:])
            bb_to_global = Rigid.cat([bb1_to_global.unsqueeze(-1), bb2_to_global.unsqueeze(-1)], dim=-1)
            bb_to_global = bb_to_global.scale_translation(self.trans_scale_factor)
            
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
                "sidechain_frames": Rigid.to_tensor_4x4(all_frames_to_global),
                "unnormalized_angles": unnormalized_angles,
                "angles": angles,
                "positions": pred_xyz,
            }

            outputs.append(preds)
            frames.append(bb_to_global.unsqueeze(-3))
        del z, z_reference_list
        
        if(_offload_inference):
            evoformer_output_dict["pair"] = (
                evoformer_output_dict["pair"].to(s.device)
            )

        outputs = dict_multimap(torch.stack, outputs)
        outputs["single"] = s
        
        outputs['frames'] = Rigid.cat(frames, dim=-3)

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
        self._init_residue_constants(frames.get_trans().dtype, frames.get_trans().device)
        return frames_and_literature_positions_to_atom23_pos(
            frames,
            nts,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
