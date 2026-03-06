# This file is part of LensKit.
# Copyright (C) 2018-2023 Boise State University.
# Copyright (C) 2023-2026 Drexel University.
# Licensed under the MIT license, see LICENSE.md for details.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass, field, replace

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
        init_scale: float = 0.1,
        user_bias: bool = True,
        item_bias: bool = True,
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

    def forward(
        self,
        user: Tensor,
        item: Tensor,
        *,
        return_norm: bool = False,
        convolution: FlexMFLightConvolution | None = None,
    ):
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

        if self.i_layers is not None and convolution is None:
            assert not return_norm
            ivec = self.i_layers[:, item, :].sum(0) * self._layer_scale
        else:
            ivec = iemb = self.i_embed(item)
            if convolution is not None:
                for layer in convolution.layers:
                    ivec = ivec + layer.item_matrix[item, :]
                ivec *= self._layer_scale

        if self.u_layers is not None and convolution is None:
            assert not return_norm
            uvec = self.u_layers[:, user, :].sum(0) * self._layer_scale
        else:
            uvec = uemb = self.u_embed(item)
            if convolution is not None:
                for layer in convolution.layers:
                    uvec = uvec + layer.user_matrix[user, :]
                uvec *= self._layer_scale

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


@dataclass
class FlexMFLightConvolution:
    """
    Data structure for the neighborhood matrix and intermediate convolution
    layers for LightGCN.
    """

    ui_matrix: torch.Tensor
    "User-item normalized adjacency matrix."
    iu_matrix: torch.Tensor
    "Item-user normalized adjacency matrix."

    layers: list[FlexMFLightConvLayer] = field(default_factory=list)
    """
    Individual embedding convolution layers.
    """

    def update(self, model: FlexMFModel, *, update_model: bool = True) -> FlexMFLightConvolution:
        """
        Compute and return updated convolution matrices based on a model's
        current parameters.

        Args:
            model:
                The model to use.
            update_model:
                If ``True``, update the model's layer matrices with the new
                layers.
        Returns:
            The updated convolution.
        """

        layers = []
        umat = model.u_embed.weight
        imat = model.i_embed.weight

        for i in range(model.layers):
            um_next = torch.mm(self.ui_matrix, imat)
            im_next = torch.mm(self.iu_matrix, umat)
            layers.append(FlexMFLightConvLayer(um_next, im_next))
            umat = um_next
            imat = im_next

        if update_model:
            model.u_layers = self.user_matrix(base=model.u_embed.weight).detach()
            model.i_layers = self.item_matrix(base=model.i_embed.weight).detach()

        return replace(self, layers=layers)

    def user_matrix(self, *, base: torch.Tensor | None = None):
        """
        Return the L x U x k matrix of user embedding convolutions.

        Args:
            base:
                The base embedding matrix.  If provided, it will be used as the
                first layer, and the returned matrix will have L+1 convolution
                layers.

        Returns:
            The full user convolution layer matrix.
        """
        layers = []
        if base is not None:
            layers.append(base)
        layers.extend(x.user_matrix for x in self.layers)
        return torch.stack(layers)

    def item_matrix(self, *, base: torch.Tensor | None = None):
        """
        Return the L x I x k matrix of item embedding convolutions.

        Args:
            base:
                The base embedding matrix.  If provided, it will be used as the
                first layer, and the returned matrix will have L+1 convolution
                layers.

        Returns:
            The full item convolution layer matrix.
        """
        layers = []
        if base is not None:
            layers.append(base)
        layers.extend(x.item_matrix for x in self.layers)
        return torch.stack(layers)


@dataclass
class FlexMFLightConvLayer:
    """
    Single layer of the LightGCN convolution network.
    """

    user_matrix: torch.Tensor
    "User embedding convolution layer matrix."
    item_matrix: torch.Tensor
    "Item embedding convolution layer matrix."
