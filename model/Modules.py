# ============================================================
# multiview_graph.py
# Cell-Gene LightGCN
# Spatial CrossView Graph Transformer
# Gate Fusion
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter

from pretrain_model import pretrain_loss


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):

    def __init__(self, dims):

        super().__init__()

        layers=[]

        for i in range(len(dims)-1):

            layers.extend([
                nn.Linear(
                    dims[i],
                    dims[i+1]
                ),
                nn.GELU()
            ])

        self.net=nn.Sequential(*layers[:-1])

    def forward(self,x):

        return self.net(x)



class ExpressionDecoder(
    nn.Module
):

    def __init__(

        self,

        dim

    ):

        super().__init__()

        self.expr=nn.Sequential(

            nn.Linear(
                dim*2,
                dim
            ),

            nn.GELU(),

            nn.Linear(
                dim,
                1
            )

        )

        self.edge=nn.Sequential(

            nn.Linear(
                dim*2,
                dim
            ),

            nn.GELU(),

            nn.Linear(
                dim,
                1
            )

        )

    def forward(

        self,

        hc,

        hg

    ):

        h=torch.cat([

            hc,

            hg

        ],-1)

        expr=F.softplus(

            self.expr(
                h
            )

        )

        edge=torch.sigmoid(

            self.edge(
                h
            )

        )

        return (

            expr.squeeze(),

            edge.squeeze()

        )

# ============================================================
# LIGHTGCN
# ============================================================

class LightGCNLayer(
    MessagePassing
):

    def __init__(self):

        super().__init__(
            aggr='mean'
        )

    def forward(
        self,
        x,
        edge_index,
        weight=None
    ):

        return self.propagate(
            edge_index,
            x=x,
            weight=weight
        )

    def message(
        self,
        x_j,
        weight=None
    ):

        if weight is None:

            return x_j

        return (
            x_j
            *
            weight.unsqueeze(-1)
        )


class CellGeneEncoder(
    nn.Module
):

    def __init__(
        self,
        n_cell,
        n_gene,
        dim,
        layers
    ):

        super().__init__()

        self.cell_emb=nn.Embedding(
            n_cell,
            dim
        )

        self.gene_emb=nn.Embedding(
            n_gene,
            dim
        )

        self.layers=nn.ModuleList([

            LightGCNLayer()

            for _ in range(
                layers
            )

        ])

        self.gene_decoder = ExpressionDecoder(dim)

    def forward(
        self,
        edge_index,
        weight=None
    ):

        x=torch.cat([

            self.cell_emb.weight,

            self.gene_emb.weight

        ])

        outs=[x]

        for layer in self.layers:

            x=layer(
                x,
                edge_index,
                weight
            )

            outs.append(
                x
            )

        x=torch.stack(
            outs
        ).mean(0)

        return x

    def decode(

            self,

            cell_id,

            gene_id,

            emb

    ):
        hc = emb[
            cell_id
        ]

        hg = emb[
            self.cell_emb.weight.shape[0]
            +
            gene_id
            ]

        return self.gene_decoder(
            hc,
            hg
        )


# ============================================================
# Sparse KNN Graph Transformer
# ============================================================

class SparseCrossViewLayer(
    nn.Module
):

    def __init__(
        self,
        dim,
        k
    ):

        super().__init__()

        self.k=k

        self.k_proj=MLP([
            dim*2,
            dim
        ])

        self.v_proj=MLP([
            dim*2,
            dim
        ])

    def forward(
        self,
        q,
        cross
    ):

        N=q.shape[0]


        kv=torch.cat([

            q,

            cross

        ],-1)

        K=self.k_proj(
            kv
        )

        V=self.v_proj(
            kv
        )

        q=F.normalize(
            q,
            dim=-1
        )

        K=F.normalize(
            K,
            dim=-1
        )

        sim=q@K.T

        idx=torch.topk(

            sim,

            self.k,

            dim=-1

        ).indices

        src=torch.arange(
            N,
            device=q.device
        )

        src=src.repeat_interleave(
            self.k
        )

        dst=idx.reshape(
            -1
        )

        attn=sim[
            src,
            dst
        ]

        attn=torch.softmax(
            attn,
            dim=0
        )

        msg=(
            V[dst]
            *
            attn.unsqueeze(
                -1
            )
        )

        out=scatter(

            msg,

            src,

            dim=0,

            reduce='sum'

        )

        return out


# =====================================================
# SINGLE CROSS LAYER
# =====================================================

class CoupledCrossLayer(
    nn.Module
):

    def __init__(
        self,
        dim,
        k
    ):

        super().__init__()

        self.spatial_layer=(
            SparseCrossViewLayer(
                dim,
                k
            )
        )

        self.expr_layer=(
            SparseCrossViewLayer(
                dim,
                k
            )
        )

    def forward(
        self,
        hs,
        he
    ):

        hs_new=(
            self.spatial_layer(
                hs,
                he
            )
        )

        he_new=(
            self.expr_layer(
                he,
                hs
            )
        )

        return (
            hs_new,
            he_new
        )




# =====================================================
# MULTI-LAYER COUPLED TRANSFORMER
# =====================================================

class CrossViewTransformer(
    nn.Module
):

    def __init__(

        self,

        dim,

        layers,

        k,
        dropout_rate = 0.2

    ):

        super().__init__()

        self.layers=nn.ModuleList([

            CoupledCrossLayer(

                dim,

                k

            )

            for _ in range(
                layers
            )

        ])

        self.dropout = nn.Dropout(dropout_rate)  # 定义 Dropout 层

    def forward(

        self,

        hs,

        he

    ):

        hs_out=[hs]

        he_out=[he]

        layer_index = 0
        for layer in self.layers:



            hs,he=layer(

                hs,

                he

            )

            if layer_index == len(self.layers)-1:
                hs = F.normalize(hs, p=2, dim=-1)
                he = F.normalize(he, p=2, dim=-1)
            else:
                hs = F.relu(hs)
                he = F.relu(he)

            # # 应用 Dropout（训练时生效，评估时自动关闭）
            # hs = self.dropout(hs)
            # he = self.dropout(he)

            hs_out.append(
                hs
            )

            he_out.append(
                he
            )

        # hs=torch.stack(
        #     hs_out).mean(0)
        #
        #
        # he=torch.stack(
        #     he_out
        # ).mean(0)

        return hs,he

# ============================================================
# Gate Fusion
# ============================================================

class GateFusion(
    nn.Module
):

    def __init__(
        self,
        dim
    ):

        super().__init__()

        self.gate=nn.Sequential(

            nn.Linear(
                dim*2,
                dim
            ),

            nn.Sigmoid()

        )

    def forward(
        self,
        h1,
        h2
    ):

        g=self.gate(

            torch.cat([

                h1,

                h2

            ],-1)

        )

        return (

            g*h1

            +

            (1-g)*h2

        )



# =====================================================
# COMMUNITY PROJECTOR
# =====================================================

class ClusterProjector(
    nn.Module
):

    def __init__(
        self,
        dim,
        proj=128
    ):

        super().__init__()

        self.project=nn.Sequential(

            nn.Linear(
                dim,
                dim
            ),

            nn.GELU(),

            nn.Linear(
                dim,
                proj
            )

        )

    def forward(
        self,
        x
    ):

        return F.normalize(

            self.project(
                x
            ),

            dim=-1

        )



def q_d(z,c,v=1):
    q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) -c, 2), 2) / v)
    q = q.pow((v + 1.0) / 2.0)
    q = (q.t() / torch.sum(q, 1)).t()
    return q
# =====================================================
# COMPLETE MODEL
# =====================================================

class MultiViewGraph(
    nn.Module
):

    def __init__(

        self,

        n_cell,

        n_gene,

        exprs_v,

        proj =6,

        dim=128,

        gcn_layers=3,

        tf_layers=4,

        k=4

    ):

        super().__init__()

        self.expr=CellGeneEncoder(

            n_cell,

            n_gene,

            dim,

            gcn_layers

        )

        self.spatial_proj=nn.Linear(
            2,
            dim
        )

        self.cross=(
            CrossViewTransformer(

                dim,

                tf_layers,

                k

            )
        )

        self.fusion=(
            GateFusion(
                dim
            )
        )

        self.projector=nn.Sequential(

            nn.Linear(
                dim,
                dim
            ),

            nn.GELU(),

            nn.Linear(
                dim,
                proj
            )

        )

        self.exprs_v = exprs_v

    def forward(

        self,

        P,

        edge_index,

        weight=None

    ):

        expr=(
            self.expr(

                edge_index,

                weight

            )
        )

        he=expr[
            :P.shape[0]
        ]

        hs=(
            self.spatial_proj(
                P
            )
        )

        hs,he=(

            self.cross(

                hs,

                he

            )

        )

        H=(

            self.fusion(

                hs,

                he

            )

        )


        pred_expr, \
            pred_edge = (
            self.expr.decode(
                self.exprs_v[1],
                self.exprs_v[2],
                expr
            )
        )

        expr_loss = pretrain_loss(

            pred_expr,

            pred_edge,

            self.exprs_v[0]

        )


        return hs, he, H,expr_loss



    def load_pretrain(

            self,

            path,
            device

    ):
        ckpt = torch.load(

            path,

            map_location=device

        )

        self.expr.load_state_dict(

            ckpt[
                "model"
            ],

            strict=False

        )

        print(
            "loaded pretrain"
        )

    def cluster_head(
            self,
            hs,
            he,
            hf
    ):
        hs = torch.sigmoid(self.projector(
            hs
        ))

        he = torch.sigmoid(self.projector(
            he
        ))

        hf = torch.sigmoid(self.projector(
            hf
        ))

        qs = q_d(hs,hf)

        qe = q_d(he,hf)



        return qs,qe