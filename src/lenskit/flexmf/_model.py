# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.linalg import norm, vecdot

from lenskit.logging import get_logger

_log = get_logger(__name__)


class FlexMFModel(nn.Module):
    """
    Torch module that defines the flexible matrix factorization model.

    Args:
        e_size:
            The size of the embedding space.
        n_users:
            The number o fusers
        n_items:
            The number of items.
        rng:
            The Torch RNG to initialize the embeddings.
        user_bias:
            Whether to learn a user bias term.
        item_bias:
            Whether to learn an item bias term.
        layers:
            The number of LightGCN layers to use (0 for no layers).
    """

    n_users: int
    n_items: int
    e_size: int
    layers: int
    _layer_scale: float

    u_bias: nn.Embedding | None = None
    i_bias: nn.Embedding | None = None
    u_embed: nn.Embedding
    i_embed: nn.Embedding

    u_layers: torch.Tensor | None = None
    i_layers: torch.Tensor | None = None

    def __init__(
        self,
        e_size: int,
        n_users: int,
        n_items: int,
        rng: torch.Generator,
        user_bias: bool,
        item_bias: bool,
        init_scale: float = 0.1,
        sparse: bool = False,
        layers: int = 0,
    ):
        super().__init__()
        self.e_size = e_size
        self.n_users = n_users
        self.n_items = n_items
        self.layers = layers
        assert layers >= 0, "layers must be nonnegative"
        self._layer_scale = 1.0 / (layers + 1)

        # user and item bias terms
        if user_bias:
            self.u_bias = nn.Embedding(n_users, 1, sparse=sparse)
        if item_bias:
            self.i_bias = nn.Embedding(n_items, 1, sparse=sparse)

        # user and item embeddings
        self.u_embed = nn.Embedding(n_users, e_size, sparse=sparse)
        self.i_embed = nn.Embedding(n_items, e_size, sparse=sparse)

        # initialize all values to a small normal
        if self.u_bias is not None:
            nn.init.normal_(self.u_bias.weight, std=init_scale, generator=rng)
        if self.i_bias is not None:
            nn.init.normal_(self.i_bias.weight, std=init_scale, generator=rng)

        nn.init.normal_(self.u_embed.weight, std=init_scale, generator=rng)
        nn.init.normal_(self.i_embed.weight, std=init_scale, generator=rng)

    @property
    def device(self):
        """
        Get the device.  FlexMF models do not support multiple simultaneous
        devices.
        """
        return self.i_embed.weight.device

    def zero_users(self, users: Tensor):
        """
        Zero the weights for the specified users (used to clear users who
        have no training interactions).
        """

        if self.u_bias is not None:
            self.u_bias.weight.data[users] = 0

        self.u_embed.weight.data[users, :] = 0

    def zero_items(self, items: Tensor):
        """
        Zero the weights for the specified items (used to clear items that
        have no training interactions).
        """

        if self.i_bias is not None:
            self.i_bias.weight.data[items] = 0

        self.i_embed.weight.data[items, :] = 0

    def update_convolution(self, ui_mat: Tensor, iu_mat: Tensor):
        """
        Update the convolution layers.
        """
        if not self.layers:
            return

        umat = self.u_embed.weight
        imat = self.i_embed.weight
        u_layers = [umat]
        i_layers = [imat]

        for _i in range(self.layers):
            um_next = torch.mm(ui_mat, imat)
            im_next = torch.mm(iu_mat, umat)
            umat = um_next
            u_layers.append(umat)
            imat = im_next
            i_layers.append(imat)

        self.u_layers = torch.stack(u_layers)
        self.i_layers = torch.stack(i_layers)

    def forward(self, user: Tensor, item: Tensor, *, return_norm: bool = False):
        """
        Matrix factorization forward pass.

        This can be applied to batches of size :math:`B` or to a set of items
        with the same user. The item tensor can also be dimensonal.

        Args:
            user:
                The user number(s), typically of size :math:`1` or :math:`B`.
            item:
                The items, typically of size :math:`B` or :math:`B \\times k`
                (to score :math:`k` items for each user).
            return_norm:
                If ``True``, return the L2 norms of the parameters affecting
                each score.  In this case, the resulting tensor has an extra
                first dimension of size 2, so ``result[0]`` is the scores and
                ``result[1]`` is the norms.
            convolution:
                A set of convolution layers to use instead of the saved ones.
        Returns:
            The scores (possibly with norms).
        """

        # look up biases and embeddings
        zero = torch.tensor(0.0)
        ub = self.u_bias(user).reshape(user.shape) if self.u_bias is not None else zero
        ib = self.i_bias(item).reshape(item.shape) if self.i_bias is not None else zero

        if self.i_layers is not None:
            ivec = self.i_layers[:, item, :].sum(0) * self._layer_scale
            if return_norm:
                iemb = self.i_layers[0, item, :]
        else:
            ivec = iemb = self.i_embed(item)

        if self.u_layers is not None:
            uvec = self.u_layers[:, user, :].sum(0) * self._layer_scale
            if return_norm:
                uemb = self.u_layers[0, user, :]
        else:
            uvec = uemb = self.u_embed(user)

        # compute the inner score
        ips = vecdot(uvec, ivec)
        score = ub + ib + ips

        if return_norm:
            l2 = torch.square(ub) + torch.square(ib) + norm(uemb, dim=-1) + norm(iemb, dim=-1)
            assert l2.shape == score.shape
            return torch.stack((score, l2))
        else:
            # we're done
            return score
