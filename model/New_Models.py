import os
import random

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
import torch.nn.functional as F

from model.AE import AE, pre_train
from model.GNN import SAGE_NET, GCN_NET
from model.Modules import MLP
from model.layers import MultiLevelGraphLayer


def community_soft_assignment(z, weight):
    """
    计算节点嵌入 z 与社区中心 weight 的余弦相似度，并通过温度参数 tau 得到软分配矩阵。
    z: (N, D)
    weight: (K, D)  社区中心质点（可学习参数）
    return: (N, K)  softmax 后的归属概率
    """
    z_norm = F.normalize(z, dim=1)
    w_norm = F.normalize(weight, dim=1)
    sim = torch.mm(z_norm, w_norm.t())
    return F.normalize(sim, p=2, dim=-1)

def similarity_from_assignment(prob):
    """
    由软分配矩阵 prob (N, K) 计算节点间的相似度图：S = prob @ prob.T
    return: (N, N)
    """
    return F.softmax(torch.mm(prob, prob.t()), dim=1)

def similarity_graph(hs, he, hf):
    """
    原始代码中 similarity_graph 的简单实现。
    这里假设 hf 是社区中心 weight，ts 是温度参数，X 是特征（可选），P 是位置（可选）。
    实际可调用 community_soft_assignment + similarity_from_assignment。
    """
    prob = community_soft_assignment(hs, hf)  # 若 hf 本身就是中心，则计算自相似图？这里保留接口，具体可根据需要修改。
    S_c = similarity_from_assignment(prob)

    prob = community_soft_assignment(he, hf)  # 若 hf 本身就是中心，则计算自相似图？这里保留接口，具体可根据需要修改。
    S_pos = similarity_from_assignment(prob)

    # ----------------
    # concat
    # ----------------

    upper=torch.cat([

        S_c,

        S_pos

    ],dim=1)

    lower=torch.cat([

        S_pos,

        S_c

    ],dim=1)

    S=torch.cat([

        upper,

        lower

    ],dim=0)



    return S

def contrast_prob(z):
    """
    计算对比概率矩阵 p_ij = exp(cos(z_i, z_j) / tau) / sum_{j} exp(...)
    z: (N, D)
    return: (N, N) 归一化概率
    """
    z_norm = F.normalize(z, dim=1)
    sim = torch.mm(z_norm, z_norm.t())
    return F.softmax(sim, dim=1)

def xclr_loss(hs, he, hf, tau=0.7, ts=0.5):
    """
    原有的 xclr_loss 保留兼容性。
    此处将其重写为支持显式三元组 (z, z_pos, z_neg) 的版本可能更直观，但保留原接口。
    下面重新实现一个更清晰的三元组损失函数。
    """
    zs = torch.cat([hs, he], dim=0)
    p = contrast_prob(zs)
    s = similarity_graph(hs, he, hf)
    loss = -(s * torch.log(p + 1e-12))
    return loss

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

# ==============================
# 新增：社区感知三元组对比损失
# ==============================
def community_triplet_loss(z, z_pos, z_neg, weight, tau_contrast=0.7, tau_community=0.5):
    """
    基于社区质心的三元组对比损失。
    参数:
        z: (N, D) anchor 嵌入
        z_pos: (N, D) 正样本嵌入
        z_neg: (N, D) 负样本嵌入
        weight: (K, D) 社区中心质点
        tau_contrast: 对比温度
        tau_community: 社区分配温度
    返回:
        loss: 标量
    """
    # 1. 计算 anchor、正、负分别对社区质心的软分配


    # zs_neg = torch.cat([z, z_neg], dim=0)
    # zs_neg2 = torch.cat([z_pos, z_neg], dim=0)
    #
    # p1 = contrast_prob(zs_neg)
    # p2 = contrast_prob(zs_neg2)
    #
    # loss_low = -(torch.log(p1 + 1e-12)).sum(dim=1).mean() - (torch.log(p2 + 1e-12)).sum(dim=1).mean()

    loss_high =xclr_loss(z, z_pos, weight).sum(dim=1).mean()

    loss = (loss_high)


    return loss.mean()

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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_sum, scatter_max


class ResMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
    def forward(self, x_raw, x):
        return self.net(x_raw) + x

class CoordGraphAttention(nn.Module):
    def __init__(self, dim, heads=4, coord_dim=2, v =1):
        super().__init__()
        self.h = heads
        self.d = dim // heads
        self.coord = nn.Sequential(nn.Linear(coord_dim, self.d), nn.GELU(), nn.Linear(self.d, self.d))
        self.wx = nn.Linear(dim, dim)
        self.att = nn.Linear(self.d * 3, 1)
        self.msg = nn.Linear(self.d * 2, self.d)
        self.alpha = nn.Parameter(torch.tensor(.1))
        self.res = ResMLP(dim)
        self.sigma = nn.Parameter(torch.tensor(.3))
        self.v = v

    def radius(
            self,
            P,
            edge
    ):
        src, dst = edge
        delta = P[dst] - P[src]
        r = torch.norm(
            delta,
            dim=-1
        )
        # local radius
        r_max, _ = (
            scatter_max(
                r,
                src,
                dim=0,
                dim_size=P.shape[0]
            )
        )

        r_max = r_max[src]

        r_max = torch.clamp(
            r_max,
            min=1e-8
        )

        rel = (delta/r_max.unsqueeze(-1))

        return (
            rel,
            r / r_max
        )

    def community_prob(self, x, cluster_layer):
        dist = ((x.unsqueeze(1) - cluster_layer).pow(2).sum(-1))
        q = (1 + dist / self.v).pow(-(self.v + 1) / 2)
        q /= q.sum(1, keepdim=True)
        return q

    def homophily(self, q, edge):
        src, dst = edge
        s = (q[src] * q[dst]).sum(-1)
        return s

    def forward(self, x, P, edge, cluster_layer = None):
        N = x.shape[0]
        x_raw = x.clone()
        src, dst = edge
        if cluster_layer is not None:
            q = self.community_prob(x,cluster_layer)
            s_uv = self.homophily(q, edge)

        rel, r = self.radius(P, edge)
        pos = self.coord(rel)
        xs = self.wx(x).view(N, self.h, self.d)
        xi = xs[src]
        xj = xs[dst]
        pos = pos.unsqueeze(1).expand(-1, self.h, -1)
        score = torch.cat([pos, xi, xj], -1)
        att = self.att(score).squeeze(-1)
        att = F.leaky_relu(att)
        att = scatter_softmax(att, src, 0)
        if cluster_layer is None:
            s_uv = scatter_softmax(torch.exp(-r.pow(2)/(2* self.sigma.pow(2)+1e-8)), src, 0)
        if s_uv.ndim == 1:
            s_uv = s_uv.unsqueeze(-1)
            s_uv = s_uv.expand(-1,self.h)

        h = self.alpha * s_uv + (1 - self.alpha) * att
        msg = torch.cat([pos, xj], -1)
        msg = self.msg(msg)
        msg = h.unsqueeze(-1) * msg
        out = scatter_sum(msg, src, dim=0, dim_size=N)
        out = out.reshape(N, -1)
        return self.res(x_raw,out),h

class GNN(nn.Module):
    def __init__(self, n_input, n_clusters, n_enc, hidden, n_z, pre_ae_epoch,X, gnn= 'sage',name =None):
        super(GNN, self).__init__()
        if(gnn == 'sage'):
            GNN_NET = SAGE_NET
        else:
            GNN_NET = GCN_NET

        n_z = n_enc
        self.n_input = n_input

        self.P_proj=MLP([
            n_input,
            n_enc
        ])

        self.fusion=(
            GateFusion(
                n_enc
            )
        )

        self.ae = AE(n_enc, hidden,
                 X.shape[1], n_enc)

        # self.hidden_gnn = nn.ModuleList([GNN_NET(n_enc, n_enc) for i in range(hidden)])
        # self.gnn_nz = GNN_NET(n_enc, n_z)

        self.hidden_gnn = nn.ModuleList([
            CoordGraphAttention(
                n_enc,
                heads=8,
                coord_dim=n_input
            )
            for _ in range(hidden)

        ])

        self.gnn_nz = CoordGraphAttention(
            n_enc,
            heads=4,
            coord_dim=n_input
        )

        self.fc1 = torch.nn.Linear(n_z, n_enc)
        self.fc2 = torch.nn.Linear(n_enc, n_enc)

        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.tau = 0.7

        self.edge_index = None

        # degree
        self.v = 1

        self.h = []

        if name is not None:
            if (not os.path.exists( f'./model/ae_pre_train_{name}.pkl')):
                pre_train(X, n_clusters, X.shape[1], n_z, n_enc, hidden, pre_ae_epoch, name=name,
                          )
            self.ae.load_state_dict(torch.load(f'./model/ae_pre_train_{name}.pkl', map_location='cuda'))

    def forward(self, X, P, edge_index, train=True):
        q = 0

        x_bar, h, z = self.ae(X)

        P_N = self.P_proj(P)
        #
        # P_x = P_N.clone()

        # P_N = self.P_proj(X)
        P_x = P_N.clone()

        x = self.fusion(P_N,z)

        x_mask = self.random_feature_mask(x,0.2)
        loop_edge_index = self.self_loop_edge_index(x_mask.shape[0]).to(device=x.device)

        # edge_index = self.adjust_edges(x,self.cluster_layer, edge_index)
        self.edge_index = edge_index

        self.h = []
        for i, layer in enumerate(self.hidden_gnn):
            x,h = layer(
                x,
                P,
                edge_index
            )
            self.h.append(h)



            # P_x = layer(
            #     P_x,
            #     P,
            #     edge_index
            # )
            #
            # x_mask = layer(
            #     x_mask,
            #     P,
            #     loop_edge_index
            #
            # )
        x,h = self.gnn_nz(x, P, edge_index)
        self.h.append(h)
        # z_pos = self.gnn_nz(P_x, edge_index)
        # z_neg = self.gnn_nz(x_mask, loop_edge_index)
        z_pos = 0
        z_neg = 0

        if(train):
            q = 1.0 / (1.0 + torch.sum(torch.pow(x.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
            q = q.pow((self.v + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()

        return x, x_bar, q, z, z_pos, z_neg

    def adjust_edges(self, h, weight, edge_index, threshold_high=0.9, threshold_low=0.05):
        """
        基于节点-社区相似度调整边结构。

        Args:
            h:         节点嵌入 (N, F)
            weight:    社区中心向量 (K, F)
            edge_index:原始边索引 (2, E)，假定为双向
            threshold_high: 添加边的相似度上界
            threshold_low:  删除边的相似度下界
        Returns:
            edge_index_new: 调整后的边索引 (2, E_new)
        """
        N = h.size(0)
        device = edge_index.device

        # 1. 计算节点-社区相似度 C: (N, K)
        norm_h = F.normalize(h, p=2, dim=-1)  # (N, F)
        norm_w = F.normalize(weight, p=2, dim=-1)  # (K, F)
        C = norm_h @ norm_w.T  # (N, K)

        # 2. 基于 C 计算节点间相似度 D: (N, N)
        norm_C = F.normalize(C, p=2, dim=-1)  # (N, K)
        D = norm_C @ norm_C.T  # (N, N)
        D.fill_diagonal_(0)  # 避免自环

        # 3. 原始邻接矩阵（布尔型）
        orig_adj = torch.zeros(N, N, dtype=torch.bool, device=device)
        orig_adj[edge_index[0], edge_index[1]] = True

        # 4. 添加/删除掩码
        add_mask = (D > threshold_high)  # 高相似度且原来无连边
        del_mask = (D < threshold_low)  # 低相似度且原来有连边

        # 5. 更新邻接矩阵
        orig_adj[add_mask] = 1
        orig_adj[del_mask] = 0

        # 6. 提取新的边索引（双向）
        edge_index_new = torch.nonzero(orig_adj).t().contiguous()  # (2, E_new)
        return edge_index_new

    def random_feature_mask(self, x, mask_ratio=0.6):
        """
        随机将特征向量中一定比例的特征置零。
        x: (N, F) 特征矩阵
        mask_ratio: 掩码比例
        return: masked_x, 相同的 edge_index (若传入)
        """
        mask = torch.rand(x.shape, device=x.device) < mask_ratio
        masked_x = x.clone()
        masked_x[mask] = 0.0
        return masked_x

    def self_loop_edge_index(self, num_nodes):
        """生成自环边索引 (2, num_nodes)"""
        row = torch.arange(num_nodes, dtype=torch.long)
        col = torch.arange(num_nodes, dtype=torch.long)
        return torch.stack([row, col], dim=0)
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def nei_con_loss(self,z1, z2, num_nodes=None, hidden_norm=True):
        """
        邻居对比损失（适配 edge_index）。
        参数:
            z1, z2: (N, D) 两个视图的嵌入
            tau: 温度系数
            edge_index: (2, E) 图边索引（无向，假设不包含自环）
            num_nodes: 节点总数（若为 None，则从 z1.shape[0] 推断）
            hidden_norm: 是否在相似度计算前对嵌入做 L2 归一化
        返回:
            loss: (N,) 每个节点的损失值（未取平均）
        """
        tau = self.tau
        edge_index= self.edge_index
        if num_nodes is None:
            num_nodes = z1.shape[0]
        z1 = self.projection(z1)
        z2 = self.projection(z2)
        # 1. 构建二值化、无自环的邻接矩阵
        adj = torch.zeros(num_nodes, num_nodes, device=z1.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        # 去除自环（若 edge_index 中可能误包含）
        adj = adj - torch.diag_embed(adj.diag())
        adj = (adj > 0).float()  # 二值化

        # 2. 计算每个节点的正样本对数（度）
        degree = adj.sum(dim=1)  # (N,)
        nei_count = degree * 2.0 + 1.0  # 自身跨视图 + 视图内邻居 + 跨视图邻居

        # 3. 相似度函数（支持 L2 归一化）
        def sim(a, b, norm):
            if norm:
                a = F.normalize(a, dim=1)
                b = F.normalize(b, dim=1)
            return torch.mm(a, b.t())  # 点积（若已归一化则为余弦相似度）

        # 4. 计算指数相似度矩阵
        f = lambda x: torch.exp(x / tau)
        intra_sim = f(sim(z1, z1, hidden_norm))  # (N, N)
        inter_sim = f(sim(z1, z2, hidden_norm))  # (N, N)

        # 5. 分子：所有正样本对的相似度之和
        pos = (inter_sim.diag()  # 自身跨视图
               + (intra_sim * adj).sum(dim=1)  # 视图内邻居
               + (inter_sim * adj).sum(dim=1))  # 跨视图邻居

        # 6. 分母：所有负样本对的相似度之和（行求和并去掉自身视图内）
        neg = (intra_sim.sum(dim=1)
               + inter_sim.sum(dim=1)
               - intra_sim.diag())  # 减去自身（避免算作负样本）

        loss = pos / (neg + 1e-12)
        loss = loss / nei_count  # 按正样本数量归一化

        return -torch.log(loss + 1e-12).mean()  # 返回每个节点的损失，通常在外部 .mean()


class GNN_OUR(nn.Module):
    def __init__(self, n_input, n_clusters, n_enc, hidden, n_z, pre_ae_epoch,X, gnn= 'sage',name =None):
        super(GNN_OUR, self).__init__()
        self.v = 1
        n_z = n_z
        self.n_input = n_input
        self.P_proj=MLP([
            n_input,
            n_enc
        ])

        self.X_proj=MLP([
            X.shape[1],
            n_enc
        ])


        self.fusion=(
            GateFusion(
                n_enc
            )
        )
        self.ae = AE(n_enc, hidden,
                 X.shape[1], n_enc)
        self.hidden_gnn = nn.ModuleList([
            CoordGraphAttention(
                n_enc,
                heads=10,
                coord_dim=n_input,
                v = self.v
            )
            for _ in range(hidden)

        ])
        self.gnn_nz = CoordGraphAttention(
            n_enc,
            heads=10,
            coord_dim=n_input,
            v=self.v
        )
        self.fc1 = torch.nn.Linear(n_z, n_enc)
        self.fc2 = torch.nn.Linear(n_enc, n_enc)
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.tau = 0.7
        self.edge_index = None
        # degree

#         self.classer = torch.nn.Linear(n_z, n_clusters)
        # self.h = []

        if name is not None:
            if (not os.path.exists( f'./model/ae_pre_train_{name}.pkl')):
                pre_train(X, n_clusters, X.shape[1], n_z, n_enc, hidden, pre_ae_epoch, name=name,
                          )
            self.ae.load_state_dict(torch.load(f'./model/ae_pre_train_{name}.pkl', map_location='cuda'))

    def forward(self, X, P, edge_index, train=True):
        q = 0

        x_bar, h, z = self.ae(X)

        # P_N = self.P_proj(P)

        # x = self.fusion(P_N,z)

        X_p = self.X_proj(X)
        x = self.fusion(X_p,z)
        # x = z

        self.edge_index = edge_index
        hs = []
        for i, layer in enumerate(self.hidden_gnn):
            x,h = layer(
                x,
                P,
                edge_index,
                cluster_layer=self.cluster_layer
            )

            hs.append(h)

        x,h = self.gnn_nz(x, P, edge_index,cluster_layer=self.cluster_layer)
        hs.append(h)

#         labels = F.softmax(
#             self.classer(x),dim=1
#         )
        labels = 0


        z_pos = 0
        z_neg = 0

        if(train):
            q = 1.0 / (1.0 + torch.sum(torch.pow(x.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
            q = q.pow((self.v + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()

        return x, x_bar, q, z, z_pos, z_neg,hs, labels

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def nei_con_loss(self, z1, z2, h, num_nodes=None, hidden_norm=True, neg_sample_num=None):
        tau = self.tau
        src, dst = self.edge_index
        if num_nodes is None:
            num_nodes = z1.shape[0]
        z1 = self.projection(z1)
        z2 = self.projection(z2)
        if hidden_norm:
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)


        # -------- 多头注意力权重处理 --------
        if h.dim() == 1:  # (E,)
            h = h.unsqueeze(1)  # (E, 1)
        n_heads = h.shape[1]  # (E, n_heads)
        # # 对每个头，在边维度上做 softmax
        # h = F.softmax(h, dim=0)  # (E, n_heads)

        # -------- 正样本（自身） --------
        self_pos = torch.exp((z1 * z2).sum(-1) / tau)  # (N,)

        # -------- 正样本（邻居） --------
        sim_intra = (z1[src] * z1[dst]).sum(-1)  # (E,)
        sim_cross = (z1[src] * z2[dst]).sum(-1)  # (E,)
        pos_intra = torch.exp(sim_intra / tau)  # (E,)
        pos_cross = torch.exp(sim_cross / tau)  # (E,)

        # 将边正样本扩展多头，并聚合到节点（对多头求和）
        pos_intra_mh = pos_intra.unsqueeze(1) * h  # (E, n_heads)
        pos_cross_mh = pos_cross.unsqueeze(1) * h
        pos_intra_node = scatter_sum(pos_intra_mh, src, dim=0, dim_size=num_nodes).sum(-1)  # (N,)
        pos_cross_node = scatter_sum(pos_cross_mh, src, dim=0, dim_size=num_nodes).sum(-1)  # (N,)

        pos = self_pos + pos_intra_node + pos_cross_node

        # -------- 负样本（排除自身和邻居） --------
        # 构建 mask：自身和邻居位置为 True（需排除）
        self_mask = torch.eye(num_nodes, dtype=torch.bool, device=z1.device)
        neighbor_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=z1.device)
        neighbor_mask[src, dst] = True
        neighbor_mask[dst, src] = True  # 若图为无向，确保双向
        exclude_mask = self_mask | neighbor_mask

        # 跨视图相似度
        sim_z1z2 = torch.mm(z1, z2.T)  # (N,N)
        # 同视图 z1 内部
        sim_z1z1 = torch.mm(z1, z1.T)  # (N,N)
        sim_z2z2 = torch.mm(z2, z2.T)  # (N,N)

        # 全量负样本（若未指定采样数量）
        if neg_sample_num is None:
            neg_cross = (torch.exp(sim_z1z2 / tau) * (~exclude_mask)).sum(-1)
            neg_same = (torch.exp(sim_z1z1 / tau) * (~exclude_mask)).sum(-1) + (torch.exp(sim_z2z2 / tau) * (~exclude_mask)).sum(-1)
            neg = neg_cross + neg_same
        else:
            # 随机采样负样本（仅从非邻居中采样）
            # 为每个节点采样 neg_sample_num 个负样本
            N = num_nodes
            all_idx = torch.arange(N, device=z1.device)
            # 预先获取每个节点的非邻居列表（但构造成本高，采用随机重试）
            # 为简便，随机生成候选对，然后过滤掉 excluded 的
            # 生成 (N * neg_sample_num) 个随机索引
            rand_idx = torch.randint(0, N, (N, neg_sample_num), device=z1.device)
            # 检查哪些对是 excluded (自身或邻居)
            # 对于每个 i，生成布尔 mask 指示是否排除
            # 这里复杂度较高，但可接受
            # 简单实现：循环（因 neg_sample_num 不大，且可用向量化）
            neg_sim_sum = torch.zeros(N, device=z1.device)
            for i in range(N):
                j = rand_idx[i]
                # 排除自身和邻居：i 的邻居列表
                neigh_i = dst[src == i]  # 这里 src 和 dst 是整体，可能包含反向
                # 但 neighbor_mask 已经构建，我们可以直接用 exclude_mask[i, j] 判断
                # 但我们需要保证采样到的 j 都不在 exclude_mask 中，所以需要重新采样直到满足
                # 简单做法：反复采样直到全部满足，但可能死循环。这里采用过滤方法：
                # 先采样多一些，然后过滤，取前 neg_sample_num 个
                # 由于代码简洁性，我们可以忽略这一复杂度，推荐使用全量方式或提供参数。
                # 实际中，可以预先计算排除掩码，然后采样时用 torch.where
            # 鉴于上述循环低效，我们采用全量方式作为默认，注释说明采样可扩展。
            # 此处为了演示，直接使用全量，但保留参数接口。
            raise NotImplementedError("Random sampling not implemented; use full neg by setting neg_sample_num=None")

        # -------- 损失 --------

        loss_main = -torch.log(pos / (neg + 1e-12))

        return loss_main.mean()

    def nei_con_loss2(self,z1, z2,h, num_nodes=None, hidden_norm=True):
        """
        邻居对比损失（适配 edge_index）。
        参数:
            z1, z2: (N, D) 两个视图的嵌入
            tau: 温度系数
            edge_index: (2, E) 图边索引（无向，假设不包含自环）
            num_nodes: 节点总数（若为 None，则从 z1.shape[0] 推断）
            hidden_norm: 是否在相似度计算前对嵌入做 L2 归一化
        返回:
            loss: (N,) 每个节点的损失值（未取平均）
        """
        tau = self.tau
        edge_index= self.edge_index
        if num_nodes is None:
            num_nodes = z1.shape[0]
        z1 = self.projection(z1)
        z2 = self.projection(z2)
        # 1. 构建二值化、无自环的邻接矩阵
        adj = torch.zeros(num_nodes, num_nodes, device=z1.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        # 去除自环（若 edge_index 中可能误包含）
        adj = adj - torch.diag_embed(adj.diag())
        adj = (adj > 0).float()  # 二值化

        # 2. 计算每个节点的正样本对数（度）
        degree = adj.sum(dim=1)  # (N,)
        nei_count = degree * 2.0 + 1.0  # 自身跨视图 + 视图内邻居 + 跨视图邻居

        # 3. 相似度函数（支持 L2 归一化）
        def sim(a, b, norm):
            if norm:
                a = F.normalize(a, dim=1)
                b = F.normalize(b, dim=1)
            return torch.mm(a, b.t())  # 点积（若已归一化则为余弦相似度）

        # 4. 计算指数相似度矩阵
        f = lambda x: torch.exp(x / tau)
        intra_sim = f(sim(z1, z1, hidden_norm))  # (N, N)
        inter_sim = f(sim(z1, z2, hidden_norm))  # (N, N)

        # 5. 分子：所有正样本对的相似度之和
        pos = (inter_sim.diag()  # 自身跨视图
               + (intra_sim * adj).sum(dim=1)  # 视图内邻居
               + (inter_sim * adj).sum(dim=1))  # 跨视图邻居

        # 6. 分母：所有负样本对的相似度之和（行求和并去掉自身视图内）
        neg = (intra_sim.sum(dim=1)
               + inter_sim.sum(dim=1)
               - intra_sim.diag())  # 减去自身（避免算作负样本）

        loss = pos / (neg + 1e-12)
        loss = loss / nei_count  # 按正样本数量归一化

        return -torch.log(loss + 1e-12).mean()  # 返回每个节点的损失，通常在外部 .mean()


class GNN_OUR_Decoder(nn.Module):
    def __init__(self, n_input, X_n, n_enc, hidden,n_clusters):
        super(GNN_OUR_Decoder, self).__init__()
        n_z = n_enc
        self.n_input = n_input
        self.v = 1

        self.hidden_gnn = nn.ModuleList([
            CoordGraphAttention(
                n_enc,
                heads=10,
                coord_dim=n_input,
                v = self.v
            )
            for _ in range(hidden)

        ])
        self.gnn_nz = CoordGraphAttention(
            n_enc,
            heads=10,
            coord_dim=n_input,
            v=self.v
        )

        self.P_proj=nn.Linear(
            n_enc,
            X_n
        )
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.edge_index = None
        # degree


    def forward(self, X, P, edge_index, train=True):
        x = X
        self.edge_index = edge_index
        for i, layer in enumerate(self.hidden_gnn):
            x,h = layer(
                x,
                P,
                edge_index,
                cluster_layer=self.cluster_layer
            )


        x,h = self.gnn_nz(x, P, edge_index,cluster_layer=self.cluster_layer)

        x = F.relu(self.P_proj(x))

        q = 1.0 / (1.0 + torch.sum(torch.pow(X.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x,q

class Transformer_GNN(nn.Module):
    def __init__(self, n_input, n_clusters, n_enc, hidden, n_z, pre_ae_epoch,X, gnn= 'sage',name =None):
        super(Transformer_GNN, self).__init__()

        self.n_input = n_input

        self.P_proj=MLP([
            n_input,
            n_enc
        ])

        self.fusion=(
            GateFusion(
                n_enc
            )
        )

        self.ae = AE(n_enc, hidden,
                 X.shape[1], n_enc)



        # Graph convolution layers
        self.convs = nn.ModuleList()

        for _ in range(hidden):
            self.convs.append(MultiLevelGraphLayer(n_enc, n_enc, 8, True))
        self.convs.append(MultiLevelGraphLayer(n_enc, n_z, 8, True))

        self.final_norm = nn.LayerNorm(n_z)



        self.fc1 = torch.nn.Linear(n_z, n_enc)
        self.fc2 = torch.nn.Linear(n_enc, n_enc)

        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self.tau = 0.7

        self.edge_index = None

        # degree
        self.v = 1

        if name is not None:
            if (not os.path.exists( f'./model/ae_pre_train_{name}.pkl')):
                pre_train(X, n_clusters, X.shape[1], n_z, n_enc, hidden, pre_ae_epoch, name=name,
                          )
            self.ae.load_state_dict(torch.load(f'./model/ae_pre_train_{name}.pkl', map_location='cuda'))

    def forward(self, X, P, edge_index, train=True):
        q = 0

        x_bar, h, z = self.ae(X)

        P_N = self.P_proj(P)
        #
        # P_x = P_N.clone()

        # P_N = self.P_proj(X)
        P_x = P_N.clone()

        x = self.fusion(P_N,z)

        x_mask = self.random_feature_mask(x,0.2)
        loop_edge_index = self.self_loop_edge_index(x_mask.shape[0]).to(device=x.device)

        # edge_index = self.adjust_edges(x,self.cluster_layer, edge_index)
        self.edge_index = edge_index


        for i, layer in enumerate(self.convs):
            x,P_x = layer(x, edge_index,P_x, edge_index)
        x = self.final_norm(x)
        z_pos = self.final_norm(P_x)
        z_neg = 0

        if(train):
            q = 1.0 / (1.0 + torch.sum(torch.pow(x.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
            q = q.pow((self.v + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()

        return x, x_bar, q, z, z_pos, z_neg

    def adjust_edges(self, h, weight, edge_index, threshold_high=0.9, threshold_low=0.05):
        """
        基于节点-社区相似度调整边结构。

        Args:
            h:         节点嵌入 (N, F)
            weight:    社区中心向量 (K, F)
            edge_index:原始边索引 (2, E)，假定为双向
            threshold_high: 添加边的相似度上界
            threshold_low:  删除边的相似度下界
        Returns:
            edge_index_new: 调整后的边索引 (2, E_new)
        """
        N = h.size(0)
        device = edge_index.device

        # 1. 计算节点-社区相似度 C: (N, K)
        norm_h = F.normalize(h, p=2, dim=-1)  # (N, F)
        norm_w = F.normalize(weight, p=2, dim=-1)  # (K, F)
        C = norm_h @ norm_w.T  # (N, K)

        # 2. 基于 C 计算节点间相似度 D: (N, N)
        norm_C = F.normalize(C, p=2, dim=-1)  # (N, K)
        D = norm_C @ norm_C.T  # (N, N)
        D.fill_diagonal_(0)  # 避免自环

        # 3. 原始邻接矩阵（布尔型）
        orig_adj = torch.zeros(N, N, dtype=torch.bool, device=device)
        orig_adj[edge_index[0], edge_index[1]] = True

        # 4. 添加/删除掩码
        add_mask = (D > threshold_high)  # 高相似度且原来无连边
        del_mask = (D < threshold_low)  # 低相似度且原来有连边

        # 5. 更新邻接矩阵
        orig_adj[add_mask] = 1
        orig_adj[del_mask] = 0

        # 6. 提取新的边索引（双向）
        edge_index_new = torch.nonzero(orig_adj).t().contiguous()  # (2, E_new)
        return edge_index_new

    def random_feature_mask(self, x, mask_ratio=0.6):
        """
        随机将特征向量中一定比例的特征置零。
        x: (N, F) 特征矩阵
        mask_ratio: 掩码比例
        return: masked_x, 相同的 edge_index (若传入)
        """
        mask = torch.rand(x.shape, device=x.device) < mask_ratio
        masked_x = x.clone()
        masked_x[mask] = 0.0
        return masked_x

    def self_loop_edge_index(self, num_nodes):
        """生成自环边索引 (2, num_nodes)"""
        row = torch.arange(num_nodes, dtype=torch.long)
        col = torch.arange(num_nodes, dtype=torch.long)
        return torch.stack([row, col], dim=0)
    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def nei_con_loss(self,z1, z2, num_nodes=None, hidden_norm=True):
        """
        邻居对比损失（适配 edge_index）。
        参数:
            z1, z2: (N, D) 两个视图的嵌入
            tau: 温度系数
            edge_index: (2, E) 图边索引（无向，假设不包含自环）
            num_nodes: 节点总数（若为 None，则从 z1.shape[0] 推断）
            hidden_norm: 是否在相似度计算前对嵌入做 L2 归一化
        返回:
            loss: (N,) 每个节点的损失值（未取平均）
        """
        tau = self.tau
        edge_index= self.edge_index
        if num_nodes is None:
            num_nodes = z1.shape[0]
        z1 = self.projection(z1)
        z2 = self.projection(z2)
        # 1. 构建二值化、无自环的邻接矩阵
        adj = torch.zeros(num_nodes, num_nodes, device=z1.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        # 去除自环（若 edge_index 中可能误包含）
        adj = adj - torch.diag_embed(adj.diag())
        adj = (adj > 0).float()  # 二值化

        # 2. 计算每个节点的正样本对数（度）
        degree = adj.sum(dim=1)  # (N,)
        nei_count = degree * 2.0 + 1.0  # 自身跨视图 + 视图内邻居 + 跨视图邻居

        # 3. 相似度函数（支持 L2 归一化）
        def sim(a, b, norm):
            if norm:
                a = F.normalize(a, dim=1)
                b = F.normalize(b, dim=1)
            return torch.mm(a, b.t())  # 点积（若已归一化则为余弦相似度）

        # 4. 计算指数相似度矩阵
        f = lambda x: torch.exp(x / tau)
        intra_sim = f(sim(z1, z1, hidden_norm))  # (N, N)
        inter_sim = f(sim(z1, z2, hidden_norm))  # (N, N)

        # 5. 分子：所有正样本对的相似度之和
        pos = (inter_sim.diag()  # 自身跨视图
               + (intra_sim * adj).sum(dim=1)  # 视图内邻居
               + (inter_sim * adj).sum(dim=1))  # 跨视图邻居

        # 6. 分母：所有负样本对的相似度之和（行求和并去掉自身视图内）
        neg = (intra_sim.sum(dim=1)
               + inter_sim.sum(dim=1)
               - intra_sim.diag())  # 减去自身（避免算作负样本）

        loss = pos / (neg + 1e-12)
        loss = loss / nei_count  # 按正样本数量归一化

        return -torch.log(loss + 1e-12).mean()  # 返回每个节点的损失，通常在外部 .mean()