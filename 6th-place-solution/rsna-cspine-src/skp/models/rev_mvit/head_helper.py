#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn

from .batchnorm_helper import (
    NaiveSyncBatchNorm1d as NaiveSyncBatchNorm1d,
)


class MLPHead(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        mlp_dim,
        num_layers,
        bn_on=False,
        bias=True,
        flatten=False,
        xavier_init=True,
        bn_sync_num=1,
        global_sync=False,
    ):
        super(MLPHead, self).__init__()
        self.flatten = flatten
        b = False if bn_on else bias
        # assert bn_on or bn_sync_num=1
        mlp_layers = [nn.Linear(dim_in, mlp_dim, bias=b)]
        mlp_layers[-1].xavier_init = xavier_init
        for i in range(1, num_layers):
            if bn_on:
                if global_sync or bn_sync_num > 1:
                    mlp_layers.append(
                        NaiveSyncBatchNorm1d(
                            num_sync_devices=bn_sync_num,
                            global_sync=global_sync,
                            num_features=mlp_dim,
                        )
                    )
                else:
                    mlp_layers.append(nn.BatchNorm1d(num_features=mlp_dim))
            mlp_layers.append(nn.ReLU(inplace=True))
            if i == num_layers - 1:
                d = dim_out
                b = bias
            else:
                d = mlp_dim
            mlp_layers.append(nn.Linear(mlp_dim, d, bias=b))
            mlp_layers[-1].xavier_init = xavier_init
        self.projection = nn.Sequential(*mlp_layers)

    def forward(self, x):
        if x.ndim == 5:
            x = x.permute((0, 2, 3, 4, 1))
        if self.flatten:
            x = x.reshape(-1, x.shape[-1])

        return self.projection(x)


class TransformerBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        dropout_rate=0.0,
        act_func="softmax",
        cfg=None,
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

        if cfg.CONTRASTIVE.NUM_MLP_LAYERS == 1:
            self.projection = nn.Linear(dim_in, num_classes, bias=True)
        else:
            self.projection = MLPHead(
                dim_in,
                num_classes,
                cfg.CONTRASTIVE.MLP_DIM,
                cfg.CONTRASTIVE.NUM_MLP_LAYERS,
                bn_on=cfg.CONTRASTIVE.BN_MLP,
                bn_sync_num=cfg.BN.NUM_SYNC_DEVICES
                if cfg.CONTRASTIVE.BN_SYNC_MLP
                else 1,
                global_sync=(
                    cfg.CONTRASTIVE.BN_SYNC_MLP and cfg.BN.GLOBAL_SYNC
                ),
            )
        self.detach_final_fc = cfg.MODEL.DETACH_FINAL_FC

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_func == "none":
            self.act = None
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        if self.detach_final_fc:
            x = x.detach()
        x = self.projection(x)

        if not self.training:
            if self.act is not None:
                x = self.act(x)
            # Performs fully convolutional inference.
            if x.ndim == 5 and x.shape[1:4] > torch.Size([1, 1, 1]):
                x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)

        return x
