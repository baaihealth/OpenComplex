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
from typing import Tuple, Optional

from opencomplex.model.primitives import Linear, LayerNorm
from opencomplex.data.data_transforms import (
    make_one_hot,
)
from opencomplex.utils.tensor_utils import add, one_hot

class RelPosEncoder(nn.Module):
    def __init__(
        self,
        c_z,
        max_relative_idx,
        max_relative_chain=0,
        use_chain_relative=False,
    ):
        super(RelPosEncoder, self).__init__()
        self.max_relative_idx = max_relative_idx
        self.max_relative_chain = max_relative_chain
        self.use_chain_relative = use_chain_relative
        self.no_bins = 2 * max_relative_idx + 1
        if max_relative_chain > 0:
            self.no_bins += 2 * max_relative_chain + 2
        if use_chain_relative:
            self.no_bins += 1

        self.linear_relpos = Linear(self.no_bins, c_z)

    def forward(self, residue_index, asym_id=None, sym_id=None, entity_id=None):
        d = residue_index[..., None] - residue_index[..., None, :]

        if asym_id is None:
            boundaries = torch.arange(
                start=-self.max_relative_idx, end=self.max_relative_idx + 1, device=d.device
            )
            reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))
            d = d[..., None] - reshaped_bins
            d = torch.abs(d)
            d = torch.argmin(d, dim=-1)
            d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
            d = d.to(residue_index.dtype)
            rel_feat = d
        else:
            rel_feats = []
            asym_id_same = torch.eq(asym_id[..., None], asym_id[..., None, :])
            offset = residue_index[..., None] - residue_index[..., None, :]

            clipped_offset = torch.clamp(
                offset + self.max_relative_idx, 0, 2 * self.max_relative_idx - 1)

            final_offset = torch.where(asym_id_same, clipped_offset,
                                    (2 * self.max_relative_idx) *
                                    torch.ones_like(clipped_offset))

            rel_pos = make_one_hot(final_offset, 2 * self.max_relative_idx + 1)
            rel_feats.append(rel_pos)

            if self.use_chain_relative:
                entity_id_same = torch.eq(entity_id[..., None], entity_id[..., None, :])
                rel_feats.append(entity_id_same.type(rel_pos.dtype)[..., None])

            if self.max_relative_chain > 0:
                rel_sym_id = sym_id[..., None] - sym_id[..., None, :]
                max_rel_chain = self.max_relative_chain

                clipped_rel_chain = torch.clamp(
                    rel_sym_id + max_rel_chain, 0, 2 * max_rel_chain)

                final_rel_chain = torch.where(entity_id_same, clipped_rel_chain,
                                            (2 * max_rel_chain + 1) *
                                            torch.ones_like(clipped_rel_chain))
                rel_chain = make_one_hot(final_rel_chain, 2 * self.max_relative_chain + 2)
                rel_feats.append(rel_chain)

            rel_feat = torch.cat(rel_feats, dim=-1)

        return self.linear_relpos(rel_feat)


class InputEmbedder(nn.Module):
    """
    Embeds a subset of the input features.

    Implements Algorithms 3 (InputEmbedder) and 4 (relpos).
    """

    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        c_z: int,
        c_m: int,
        max_relative_idx,
        max_relative_chain=0,
        use_chain_relative=False,
        **kwargs,
    ):
        """
        Args:
            tf_dim:
                Final dimension of the target features
            msa_dim:
                Final dimension of the MSA features
            c_z:
                Pair embedding dimension
            c_m:
                MSA embedding dimension
        """
        super(InputEmbedder, self).__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.c_z = c_z
        self.c_m = c_m

        self.linear_tf_z_i = Linear(tf_dim, c_z)
        self.linear_tf_z_j = Linear(tf_dim, c_z)
        self.linear_tf_m = Linear(tf_dim, c_m)
        self.linear_msa_m = Linear(msa_dim, c_m)

        self.rpe = RelPosEncoder(c_z, max_relative_idx, max_relative_chain, use_chain_relative)


    def forward(
        self,
        tf: torch.Tensor,
        ri: torch.Tensor,
        msa: torch.Tensor,
        inplace_safe: bool = False,
        asym_id: torch.Tensor = None,
        sym_id: torch.Tensor = None,
        entity_id: torch.Tensor = None,
        hmm: torch.Tensor = None,
        secondary_structure: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tf:
                "target_feat" features of shape [*, N_res, tf_dim]
            ri:
                "residue_index" features of shape [*, N_res]
            msa:
                "msa_feat" features of shape [*, N_clust, N_res, msa_dim]
        Returns:
            msa_emb:
                [*, N_clust, N_res, C_m] MSA embedding
            pair_emb:
                [*, N_res, N_res, C_z] pair embedding

        """
        # [*, N_res, c_z]
        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)

        # [*, N_res, N_res, c_z]
        pair_emb = self.rpe(ri.type(tf_emb_i.dtype), asym_id, sym_id, entity_id)
        pair_emb = add(pair_emb,
            tf_emb_i[..., None, :],
            inplace=inplace_safe
        )
        pair_emb = add(pair_emb,
            tf_emb_j[..., None, :, :],
            inplace=inplace_safe
        )

        # [*, N_clust, N_res, c_m]
        n_clust = msa.shape[-3]
        tf_m = (
            self.linear_tf_m(tf)
            .unsqueeze(-3)
            .expand(((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1)))
        )
        msa_emb = self.linear_msa_m(msa) + tf_m

        return msa_emb, pair_emb


class RecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """
    def __init__(
        self,
        c_m: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        inf: float = 1e8,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            no_bins:
                Number of distogram bins
        """
        super(RecyclingEmbedder, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = inf

        self.linear = Linear(self.no_bins, self.c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted C_beta coordinates
        Returns:
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        # [*, N, C_m]
        m_update = self.layer_norm_m(m)
        if(inplace_safe):
            m.copy_(m_update)
            m_update = m

        # [*, N, N, C_z]
        z_update = self.layer_norm_z(z)
        if(inplace_safe):
            z.copy_(z_update)
            z_update = z

        # This squared method might become problematic in FP16 mode.
        bins = torch.linspace(
            self.min_bin,
            self.max_bin,
            self.no_bins,
            dtype=x.dtype,
            device=x.device,
            requires_grad=False,
        )
        squared_bins = bins ** 2
        upper = torch.cat(
            [squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1
        )
        d = torch.sum(
            (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True
        )

        # [*, N, N, no_bins]
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)

        # [*, N, N, C_z]
        d = self.linear(d)
        z_update = add(z_update, d, inplace_safe)

        return m_update, z_update


class TemplateAngleEmbedder(nn.Module):
    """
    Embeds the "template_angle_feat" feature.

    Implements Algorithm 2, line 7.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:
                Final dimension of "template_angle_feat"
            c_out:
                Output channel dimension
        """
        super(TemplateAngleEmbedder, self).__init__()

        self.c_out = c_out
        self.c_in = c_in

        self.linear_1 = Linear(self.c_in, self.c_out, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.c_out, self.c_out, init="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, N_templ, N_res, c_in] "template_angle_feat" features
        Returns:
            x: [*, N_templ, N_res, C_out] embedding
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        return x


class TemplatePairEmbedder(nn.Module):
    """
    Embeds "template_pair_feat" features.

    Implements Algorithm 2, line 9.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:

            c_out:
                Output channel dimension
        """
        super(TemplatePairEmbedder, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        # Despite there being no relu nearby, the source uses that initializer
        self.linear = Linear(self.c_in, self.c_out, init="relu")

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, C_in] input tensor
        Returns:
            [*, C_out] output tensor
        """
        x = self.linear(x)

        return x


class ExtraMSAEmbedder(nn.Module):
    """
    Embeds unclustered MSA sequences.

    Implements Algorithm 2, line 15
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        **kwargs,
    ):
        """
        Args:
            c_in:
                Input channel dimension
            c_out:
                Output channel dimension
        """
        super(ExtraMSAEmbedder, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.linear = Linear(self.c_in, self.c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_extra_seq, N_res, C_in] "extra_msa_feat" features
        Returns:
            [*, N_extra_seq, N_res, C_out] embedding
        """
        x = self.linear(x)

        return x

class HMMEmbedder(nn.Module):
    def __init__(self, c_in, c_m):
        super(HMMEmbedder,self).__init__()
        self.linear_hmm = Linear(c_in, c_m)

    def forward(self, hmm):
        return self.linear_hmm(hmm)

class SecondaryStructureEmbedder(nn.Module):
    def __init__(self, c_in, c_z):
        super(SecondaryStructureEmbedder,self).__init__()
        self.linear_ss = Linear(c_in, c_z)

    def forward(self, ss):
        return self.linear_ss(ss)