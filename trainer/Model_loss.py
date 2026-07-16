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

    p = torch.exp(sim / tau)
    p = p * (1 - torch.eye(p.size(0), device=p.device))  # 非原地
    p = p / p.sum(-1, keepdim=True)

    return p


def sim_loss(x, x_aug, temperature, sym=True):
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / temperature)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    if sym:
        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

        loss_0 = - torch.log(loss_0).mean()
        loss_1 = - torch.log(loss_1).mean()
        loss = (loss_0 + loss_1) / 2.0
        return loss
    else:
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss_1 = - torch.log(loss_1).mean()
        return loss_1

def xclr_loss(

    hs,
    he,
    hf,
    P,
    X,
    tau=.7,

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

        b=.5,

        c= 0,

        d=.1

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

        # lx = sim_loss(hs,he,0.7)

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

            self.d*expr_loss['loss']

        )

        return {

            "loss":total,

            "cluster":lc,

            "xclr":lx
#            "graph":lg
        }