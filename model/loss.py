# ============================================================
# losses.py
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# STUDENT-T CLUSTER
# ============================================================

class ClusterHead(nn.Module):

    def __init__(
        self,
        dim,
        K
    ):

        super().__init__()

        self.center=nn.Parameter(
            torch.randn(
                K,
                dim
            )
        )

    def forward(
        self,
        z
    ):

        d=torch.cdist(
            z,
            self.center
        )

        q=(
            1+
            d.pow(2)
        ).pow(-1)

        q=q/q.sum(
            -1,
            keepdim=True
        )

        return q


# ============================================================
# CLUSTER ALIGNMENT
# ============================================================

def cluster_loss(
    qs,
    qe
):

    m=(qs+qe)/2

    return (

        F.kl_div(
            qs.log(),
            m,
            reduction='batchmean'
        )

        +

        F.kl_div(
            qe.log(),
            m,
            reduction='batchmean'
        )

    )/2


# ============================================================
# XCLR
# ============================================================

def similarity_graph(
    z,
    ts=0.5
):

    z=F.normalize(
        z,
        dim=-1
    )

    g=z@z.T

    s=F.softmax(
        g/ts,
        dim=-1
    )

    return s


def contrast_prob(
    z,
    tau=0.2
):

    z=F.normalize(
        z,
        dim=-1
    )

    sim=z@z.T

    p=torch.exp(
        sim/tau
    )

    p.fill_diagonal_(0)

    p=(
        p
        /
        p.sum(
            -1,
            keepdim=True
        )
    )

    return p


def xclr_loss(

    hs,

    he,

    tau=.2,

    ts=.5

):

    zs=torch.cat([

        hs,

        he

    ])

    p=contrast_prob(

        zs,

        tau

    )

    s=similarity_graph(

        zs,

        ts

    )

    loss=-(

        s

        *

        torch.log(

            p+1e-12

        )

    ).sum(-1)

    return loss.mean()


# ============================================================
# GRAPH REGULARIZATION
# ============================================================

def graph_reg(

    H,

    edge

):

    src,dst=edge

    diff=(

        H[src]

        -

        H[dst]

    )

    return (

        diff
        .pow(2)
        .sum(-1)
        .mean()

    )


# ============================================================
# TOTAL LOSS
# ============================================================

class TotalLoss(
    nn.Module
):

    def __init__(

        self,

        dim,

        K=20,

        a=.3,

        b=.3,

        c=.2,

        d=.2

    ):

        super().__init__()

        self.cluster=ClusterHead(
            dim,
            K
        )

        self.a=a
        self.b=b
        self.c=c
        self.d=d

    def forward(

        self,

        hs,

        he,

        edge_s,

        edge_e,

        expr_loss

    ):

        qs=self.cluster(
            hs
        )

        qe=self.cluster(
            he
        )

        lc=cluster_loss(
            qs,
            qe
        )

        lx=xclr_loss(
            hs,
            he
        )

        lg=(

            graph_reg(
                hs,
                edge_s
            )

            +

            graph_reg(
                he,
                edge_e
            )

        )

        total=(

            self.a*lc

            +

            self.b*lx

            +

            self.c*lg

            +

            self.d*expr_loss

        )

        return {

            "loss":total,

            "cluster":lc,

            "xclr":lx,

            "graph":lg

        }