# =====================================
# losses/pretrain_loss.py
# =====================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpressionLoss(
    nn.Module
):

    def __init__(

        self,

        alpha=1.0,

        beta=1.0

    ):

        super().__init__()

        self.alpha=alpha
        self.beta=beta

    def forward(

        self,

        pred_expr,

        pred_edge,

        target

    ):

        expr_target=torch.log1p(
            target
        )

        edge_target=(
            target>0
        ).float()

        expr_loss=F.mse_loss(

            pred_expr,

            expr_target

        )

        edge_loss=(
            F.binary_cross_entropy(

                pred_edge,

                edge_target

            )
        )

        total=(
            self.alpha
            *
            expr_loss
            +
            self.beta
            *
            edge_loss
        )

        return {

            "loss":total,

            "expr":expr_loss,

            "edge":edge_loss

        }