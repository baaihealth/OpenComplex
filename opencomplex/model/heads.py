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
import torch.nn.functional as F

from opencomplex.model.primitives import Linear, LayerNorm
from opencomplex.utils.loss import (
    compute_plddt,
    compute_tm,
    compute_predicted_aligned_error,
)


class AuxiliaryHeads(nn.Module):
    def __init__(self, config):
        super(AuxiliaryHeads, self).__init__()

        self.plddt = PerResidueLDDTCaPredictor(
            **config["lddt"],
        )

        self.distogram = DistogramHead(
            **config["distogram"],
        )

        self.masked_msa = MaskedMSAHead(
            **config["masked_msa"],
        )

        self.experimentally_resolved = ExperimentallyResolvedHead(
            **config["experimentally_resolved"],
        )

        if config.tm.enabled:
            self.tm = TMScoreHead(
                **config.tm,
            )

        self.config = config

    def forward(self, outputs, asym_id=None, interface=False):
        aux_out = {}
        lddt_logits = self.plddt(outputs["sm"]["single"])
        aux_out["lddt_logits"] = lddt_logits

        # Required for relaxation later on
        aux_out["plddt"] = compute_plddt(lddt_logits)

        distogram_logits = self.distogram(outputs["pair"])
        aux_out["distogram_logits"] = distogram_logits

        masked_msa_logits = self.masked_msa(outputs["msa"])
        aux_out["masked_msa_logits"] = masked_msa_logits

        experimentally_resolved_logits = self.experimentally_resolved(
            outputs["single"]
        )
        aux_out[
            "experimentally_resolved_logits"
        ] = experimentally_resolved_logits

        if self.config.tm.enabled:
            tm_logits = self.tm(outputs["pair"])
            aux_out["tm_logits"] = tm_logits
            ptm = compute_tm(tm_logits, asym_id=None, interface=False, **self.config.tm)
            if interface:
                iptm = compute_tm(
                    tm_logits, asym_id=asym_id, interface=interface, **self.config.tm
                )
                ptm = ptm * 0.2 + iptm * 0.8
            aux_out["predicted_tm_score"] = ptm
            aux_out.update(
                compute_predicted_aligned_error(
                    tm_logits,
                    **self.config.tm,
                )
            )

        return aux_out


class PerResidueLDDTCaPredictor(nn.Module):
    def __init__(self, no_bins, c_in, c_hidden):
        super(PerResidueLDDTCaPredictor, self).__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.layer_norm = LayerNorm(self.c_in)

        self.linear_1 = Linear(self.c_in, self.c_hidden, init="relu")
        self.linear_2 = Linear(self.c_hidden, self.c_hidden, init="relu")
        self.linear_3 = Linear(self.c_hidden, self.no_bins, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s = self.layer_norm(s)
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        return s


class DistogramHead(nn.Module):
    """
    Computes a distogram probability distribution.

    For use in computation of distogram loss, subsection 1.9.8
    """

    def __init__(self, c_z, no_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of distogram bins
        """
        super(DistogramHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(self.c_z, self.no_bins, init="final")

    def forward(self, z):  # [*, N, N, C_z]
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N, N, no_bins] distogram probability distribution
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        logits = logits + logits.transpose(-2, -3)
        return logits


class TMScoreHead(nn.Module):
    """
    For use in computation of TM-score, subsection 1.9.7
    """

    def __init__(self, c_z, no_bins, **kwargs):
        """
        Args:
            c_z:
                Input channel dimension
            no_bins:
                Number of bins
        """
        super(TMScoreHead, self).__init__()

        self.c_z = c_z
        self.no_bins = no_bins

        self.linear = Linear(self.c_z, self.no_bins, init="final")

    def forward(self, z):
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pairwise embedding
        Returns:
            [*, N_res, N_res, no_bins] prediction
        """
        # [*, N, N, no_bins]
        logits = self.linear(z)
        return logits


class MaskedMSAHead(nn.Module):
    """
    For use in computation of masked MSA loss, subsection 1.9.9
    """

    def __init__(self, c_m, c_out, **kwargs):
        """
        Args:
            c_m:
                MSA channel dimension
            c_out:
                Output channel dimension
        """
        super(MaskedMSAHead, self).__init__()

        self.c_m = c_m
        self.c_out = c_out

        self.linear = Linear(self.c_m, self.c_out, init="final")

    def forward(self, m):
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
        Returns:
            [*, N_seq, N_res, C_out] reconstruction
        """
        # [*, N_seq, N_res, C_out]
        logits = self.linear(m)
        return logits


class ExperimentallyResolvedHead(nn.Module):
    """
    For use in computation of "experimentally resolved" loss, subsection
    1.9.10
    """

    def __init__(self, c_s, c_out, **kwargs):
        """
        Args:
            c_s:
                Input channel dimension
            c_out:
                Number of distogram bins
        """
        super(ExperimentallyResolvedHead, self).__init__()

        self.c_s = c_s
        self.c_out = c_out

        self.linear = Linear(self.c_s, self.c_out, init="final")

    def forward(self, s):
        """
        Args:
            s:
                [*, N_res, C_s] single embedding
        Returns:
            [*, N, C_out] logits
        """
        # [*, N, C_out]
        logits = self.linear(s)
        return logits


class AuxiliaryHeadsRNA(nn.Module):
    def __init__(self, config):
        super(AuxiliaryHeadsRNA, self).__init__()

        self.geometry = Geometry_Head(**config["geometry_head"])
        self.torsion = Torsion_Head(**config["torsion_head"])
        self.mask_msa = MaskedMSAHead(**config["masked_msa"])

        self.config = config

    def forward(self, outputs):
        aux_out = {}

        aux_out["geometry_head"] = self.geometry(outputs["pair"])
        aux_out["torsion_head"] = self.torsion(outputs["single"])
        aux_out["masked_msa_logits"] = self.mask_msa(outputs["msa"])

        return aux_out


class Angle_Block(nn.Module):
    def __init__(self, hidden_dim):
        super(Angle_Block, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear_1 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.linear_2 = nn.Linear(self.hidden_dim,self.hidden_dim)

    def forward(self, seq_embed):
        seq_embed = self.linear_1(F.relu(seq_embed))
        seq_embed = self.linear_2(F.relu(seq_embed))
        
        return seq_embed

class Torsion_Head(nn.Module):
    def __init__(self, c_s, hidden_dim, no_blocks, no_angles, angle_bins):
        super(Torsion_Head,self).__init__()
        self.seq_dim = c_s
        self.hidden_dim = hidden_dim
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.angle_bins = angle_bins
        
        self.linear_in = nn.Linear(self.seq_dim,self.hidden_dim)
        self.angle_layer = Angle_Block(self.hidden_dim)
        self.linear_out_dis = nn.Linear(self.hidden_dim,self.no_angles*self.angle_bins)

    def forward(self, seq_embed):
        B, N, embed_dim = seq_embed.shape
        seq_embed = self.linear_in(F.relu(seq_embed))

        for i in range(self.no_blocks):
            seq_embed = seq_embed + self.angle_layer(seq_embed)

        seq_embed = F.relu(seq_embed)

        angles_dis = self.linear_out_dis(seq_embed)
        angles_dis = F.log_softmax(angles_dis.contiguous().view(B, N, self.no_angles, self.angle_bins),dim=-1)

        output = {}
        output["angles_dis"] = angles_dis 

        return output

class Geometry_Head(nn.Module):
    def __init__(self, c_z, dist_bins, omega_bins, theta_bins, phi_bins):
        super(Geometry_Head,self).__init__()
        self.pair_dim = c_z
        self.dis_bins = dist_bins
        self.omg_bins = omega_bins
        self.theta_bins = theta_bins
        self.phi_bins = phi_bins

        self.linear_dis_n = nn.Linear(self.pair_dim,self.dis_bins)
        self.linear_dis_c4 = nn.Linear(self.pair_dim,self.dis_bins)
        self.linear_dis_p = nn.Linear(self.pair_dim,self.dis_bins)
        self.linear_omg = nn.Linear(self.pair_dim,self.omg_bins)
        self.linear_theta = nn.Linear(self.pair_dim,self.theta_bins)
        # self.linear_phi = nn.Linear(self.pair_dim,self.phi_bins)

    def forward(self, pair_embed):
        pred_dis_n = self.linear_dis_n(pair_embed)
        pred_dis_n = F.log_softmax(pred_dis_n + pred_dis_n.permute(0,2,1,3),dim=-1)
        pred_dis_c4 = self.linear_dis_c4(pair_embed)
        pred_dis_c4 = F.log_softmax(pred_dis_c4 + pred_dis_c4.permute(0,2,1,3),dim=-1)
        pred_dis_p = self.linear_dis_p(pair_embed)
        pred_dis_p = F.log_softmax(pred_dis_p + pred_dis_p.permute(0,2,1,3),dim=-1)
        pred_omg = self.linear_omg(pair_embed)
        pred_omg = F.log_softmax(pred_omg + pred_omg.permute(0,2,1,3),dim=-1)
        pred_theta = F.log_softmax(self.linear_theta(pair_embed),dim=-1)
        # pred_phi = F.log_softmax(self.linear_phi(pair_embed),dim=-1)

        output = {}
        output["pred_dis_n"] = pred_dis_n
        output["pred_dis_c4"] = pred_dis_c4
        output["pred_dis_p"] = pred_dis_p
        output["pred_omg"] = pred_omg
        output["pred_theta"] = pred_theta
        # output["pred_phi"] = pred_phi

        return output