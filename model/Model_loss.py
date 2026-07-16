# ============================================================
# losses.py
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def cluster_loss(
    qs,
    qe
):

    ps = target_distribution(qs)
    pe = target_distribution(qe)

    return (

        F.kl_div(
            qs.log(),
            ps,
            reduction='batchmean'
        )

        +

        F.kl_div(
            qe.log(),
            pe,
            reduction='batchmean'
        )

    )/2


# ============================================================
# XCLR
# ============================================================

# =====================================================
# BUILD SOFT GRAPH
# =====================================================

def cosine_graph(
    z,
    temp
):

    z=F.normalize(
        z,
        dim=-1
    )

    sim=z@z.T

    sim=F.softmax(

        sim/temp,

        dim=-1

    )

    return sim


# =====================================================
# BUILD 2N×2N GRAPH
# =====================================================

def similarity_graph(

    P,

    X,

    hf,

    ts

):

    # ----------------
    # spatial block
    # ----------------

    Sss=cosine_graph(

        P,

        ts

    )

    # ----------------
    # expression block
    # ----------------

    See=cosine_graph(

        X,

        ts

    )

    # ----------------
    # cross block
    # ----------------

    Sse=cosine_graph(

        hf,

        ts

    )

    Ses=Sse.T

    # ----------------
    # concat
    # ----------------

    upper=torch.cat([

        Sss,

        Sse

    ],dim=1)

    lower=torch.cat([

        Ses,

        See

    ],dim=1)

    S=torch.cat([

        upper,

        lower

    ],dim=0)

    return S


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
    hf,
    P,
    X,
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

    # soft relation

    s = (

        similarity_graph(

            P,

            X,

            hf,

            ts

        )

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

        model,

        a=.4,

        b=.4,

        c= 0,

        d=.2

    ):

        super().__init__()

        self.model = model
        self.a=a
        self.b=b
        self.c=c
        self.d=d

    def forward(
        self,
        hs,
        he,
        hf,
        P,
        X,
        expr_loss,
        edge_s = None,
        edge_e = None,

    ):

        qs,qe= self.model.cluster_head(
            hs,he,hf
        )

        lc=cluster_loss(
            qs,
            qe
        )

        lx=xclr_loss(
            hs,
            he,
            hf,
            P,
            X
        )

        # lg=(
        #
        #     graph_reg(
        #         hs,
        #         edge_s
        #     )
        #
        #     +
        #
        #     graph_reg(
        #         he,
        #         edge_e
        #     )
        #
        # )
        #     +
        #
        #     self.c*lg

        total=(

            self.a*lc

            +

            self.b*lx

            +

            self.d*expr_loss

        )

        return {

            "loss":total,

            "cluster":lc,

            "xclr":lx
#            "graph":lg
        }