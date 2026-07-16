# ==========================================================
# main.py
# ==========================================================

import os
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data

from Clusterutils import evaluate_embeddings_cluster
from model.Modules_F import (
    CellGeneEncoder,
    MultiViewGraph
)
from model.New_Models import GNN, community_triplet_loss, Transformer_GNN, GNN_OUR
from model.P_encoder import SpatialFeatureEncoder

from pretrain_model import (
    pretrain, create_loader, pretrain_loss
)
from queryopt import pyabcore

from trainer.Model_loss import (
    TotalLoss, target_distribution
)

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader

from trainer.engine import (
    train,
    evaluate
)
from 可视化 import cluster_and_compare


def get_coredata(X, edge_index,device):
    n_cells, n_genes = X.shape
    # 1. 提取所有非零边
    row, col = torch.nonzero(X > 0, as_tuple=True)      # 细胞索引, 基因索引
    # 2. 构建正向边 (细胞→基因) 和反向边 (基因→细胞)
    #    基因节点添加偏移 n_cells，使其索引与细胞不重叠
    edge_index = torch.stack([row, col], dim=1).int().cpu().numpy()         # (2, E)  # (2, E)
    # print('build index')
    abcore = pyabcore.Pyabcore(X.shape[0], X.shape[1])
    # start_time = time()
    abcore.index(edge_index)
    # index_time = time()
    # print('finished, time:{}'.format(index_time - start_time))
    a = 2
    b = 1
    core_u_x = torch.BoolTensor([]).to(device)
    core_i_x = torch.BoolTensor([]).to(device)
    while 1:
        abcore.query(a, b)
        result_u = torch.BoolTensor(abcore.get_left()).to(device)
        result_i = torch.BoolTensor(abcore.get_right()).to(device)
        if(result_i.sum() < len(result_i)*0.01):
            print('max b:{}'.format(b-1))
            max_b = b-1
            break

        core_u_x = torch.cat((core_u_x, result_u.unsqueeze(-1)),dim=1)
        core_i_x = torch.cat((core_i_x, result_i.unsqueeze(-1)),dim=1)
        b += 1
    X = torch.cat((X,core_u_x),dim=1)
    return X, core_u_x, core_i_x,max_b

# =========================
# LOAD
# =========================
def load_group(path ,name):
    if name == 'a':
        X = np.load(os.path.join(path, "X_tensor.npy"))      # (P, N, G)
        P = np.load(os.path.join(path, "P_tensor.npy"))      # (P, N, 2)
        Xc = np.load(os.path.join(path, "X_cell_tensor.npy"))    # (P, N, F)
        L_c = Xc[:,-1].astype(int)
        Xc = Xc[:,:-1]
        Xgene = np.load(os.path.join(path, "X_gene.npy"))        # (G, Fg)
        Pn = np.load(os.path.join(path, "cell_patch.npy"))
        N, G = X[Pn==4].shape
        # _, _, F = Xc.shape
        # X = X.reshape(-1, G)
        # P = P.reshape(-1, 2)
        # Xc = Xc.reshape(-1, F)
        return X[Pn==4], P[Pn==4], Pn,N, G, Xc[Pn==4],Xgene,L_c[Pn==4]
    else:
        X = np.load(os.path.join(path, "X_tensor.npy"))      # (P, N, G)
        P = np.load(os.path.join(path, "P_tensor.npy"))      # (P, N, 2)
        Xc = np.load(os.path.join(path, "X_cell_tensor.npy"))    # (P, N, F)
        L_c = Xc.astype(int)
        Xc = Xc[:,:-1]
        Xgene = np.load(os.path.join(path, "X_gene.npy"))        # (G, Fg)
        Pn = np.load(os.path.join(path, "cell_patch.npy"))
        N, G = X.shape
        # _, _, F = Xc.shape
        # X = X.reshape(-1, G)
        # P = P.reshape(-1, 2)
        # Xc = Xc.reshape(-1, F)
        return X, P, Pn,N, G, Xc,Xgene,L_c



def kmeans(x, ncluster, niter=10):
    '''
    x : torch.tensor(data_num,data_dim)
    ncluster : The number of clustering for data_num
    niter : Number of iterations for kmeans
    '''
    N, D = x.size()
    c = x[torch.randperm(N)[:ncluster]] # init clusters at random
    for i in range(niter):
        a = ((x[:, None, :] - c[None, :, :])**2).sum(-1).argmin(1)
        c = torch.stack([x[a==k].mean(0) for k in range(ncluster)])
        nanix = torch.any(torch.isnan(c), dim=1)
        ndead = nanix.sum().item()
        c[nanix] = x[torch.randperm(N)[:ndead]]
    return c

# =========================
# KNN + SYMMETRIC NORMALIZATION (undirected, GPU)
# =========================
def build_spatial_graph(pos, X=None, k=100):
    """
    Build undirected KNN graph with symmetric normalization weights.
    If X is provided, edges are the intersection of KNN graphs from pos and X.
    If X is None, only uses pos (original behavior).
    pos: (N, d1) tensor on DEVICE
    X:   (N, d2) tensor on DEVICE (optional)
    Returns:
        edge_index: (2, E) undirected edges (each edge appears once, u<v)
        edge_weight: (E,)  D^{-1/2} A D^{-1/2}
    """
    N = pos.size(0)
    device = pos.device

    # ------------- 情况1：仅用 pos -------------
    if X is None:
        dist = torch.cdist(pos, pos)
        _, nn_idx = torch.topk(dist, k=k, dim=1, largest=False)
        row = torch.arange(N, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
        col = nn_idx.reshape(-1)
        edge_index_dir = torch.stack([row, col], dim=0)
        edge_index_undir = torch.cat([edge_index_dir, edge_index_dir.flip(0)], dim=1)
        edge_index_undir = torch.unique(edge_index_undir, dim=1)
        u, v = edge_index_undir
        deg = torch.zeros(N, device=device)
        deg.index_add_(0, u, torch.ones(u.size(0), device=device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg == 0] = 0.0
        weight = deg_inv_sqrt[u] * deg_inv_sqrt[v]
        return edge_index_undir, weight

    # ------------- 情况2：pos 和 X 的交集 -------------
    # 1. pos 的 KNN 图
    dist_pos = torch.cdist(pos, pos)
    _, nn_idx_pos = torch.topk(dist_pos, k=k, dim=1, largest=False)
    row_pos = torch.arange(N, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
    col_pos = nn_idx_pos.reshape(-1)
    edge_dir_pos = torch.stack([row_pos, col_pos], dim=0)          # (2, N*k)
    edge_undir_pos = torch.sort(edge_dir_pos, dim=0)[0]            # 排序使 u<v
    edge_undir_pos = torch.unique(edge_undir_pos, dim=1)           # (2, E_pos)

    # 2. X 的 KNN 图
    dist_X = torch.cdist(X, X)
    _, nn_idx_X = torch.topk(dist_X, k=k, dim=1, largest=False)
    row_X = torch.arange(N, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
    col_X = nn_idx_X.reshape(-1)
    edge_dir_X = torch.stack([row_X, col_X], dim=0)
    edge_undir_X = torch.sort(edge_dir_X, dim=0)[0]
    edge_undir_X = torch.unique(edge_undir_X, dim=1)               # (2, E_X)

    # 3. 取交集（编码为 64 位整数）
    codes_pos = edge_undir_pos[0] * N + edge_undir_pos[1]          # 因为已排序，u*N+v 唯一
    codes_X   = edge_undir_X[0] * N + edge_undir_X[1]
    mask = torch.isin(codes_pos, codes_X)
    codes_common = codes_pos[mask]

    if codes_common.numel() == 0:
        # 无公共边，返回空
        return torch.empty((2, 0), dtype=torch.long, device=device), torch.empty(0, device=device)

    # 4. 解码得到公共无向边
    u = codes_common // N
    v = codes_common % N
    edge_index_undir = torch.stack([u, v], dim=0)                  # (2, E)

    # 5. 对称归一化
    all_nodes = torch.cat([u, v])
    deg = torch.zeros(N, device=device)
    deg.index_add_(0, all_nodes, torch.ones(all_nodes.size(0), device=device))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg == 0] = 0.0
    weight = deg_inv_sqrt[u] * deg_inv_sqrt[v]

    return edge_index_undir, weight


def build_expression_graph(X):
    """
    构建带对称归一化权重的无向二分图（细胞‑基因）。
    X: (n_cells, n_genes) tensor on DEVICE, 表达矩阵（一般需先进行非负处理）
    Returns:
        edge_index: (2, 2*E)  包含正向(细胞→基因)和反向(基因→细胞)的边
        edge_weight: (2*E,)   对应边的对称归一化权重
    """
    n_cells, n_genes = X.shape
    # 1. 提取所有非零边
    row, col = torch.nonzero(X > 0, as_tuple=True)      # 细胞索引, 基因索引
    w = X[row, col]                                     # 原始表达值作为权重

    # 2. 构建正向边 (细胞→基因) 和反向边 (基因→细胞)
    #    基因节点添加偏移 n_cells，使其索引与细胞不重叠
    pos_edge = torch.stack([row, col + n_cells], dim=0)        # (2, E)
    neg_edge = torch.stack([col + n_cells, row], dim=0)        # (2, E)
    edge_index = torch.cat([pos_edge, neg_edge], dim=1)        # (2, 2E)

    # 3. 计算对称归一化权重  D^{-1/2} A D^{-1/2}
    #    度向量: 对细胞节点，度为行和；对基因节点，度为列和
    deg_cell = X.sum(dim=1) + 1e-8       # (n_cells,)
    deg_gene = X.sum(dim=0) + 1e-8       # (n_genes,)
    # 每条边的归一化因子 = w / sqrt(deg_cell_u * deg_gene_v)
    deg_cell_inv_sqrt = deg_cell.pow(-0.5)
    deg_gene_inv_sqrt = deg_gene.pow(-0.5)

    w_norm = w * deg_cell_inv_sqrt[row] * deg_gene_inv_sqrt[col]   # (E,)

    # 正向边和反向边使用相同的归一化权重
    edge_weight = torch.cat([w_norm, w_norm], dim=0)               # (2E,)

    return edge_index, edge_weight


import torch


def edge_v(expr, edge_index, N):
    """
    expr: (n_cells, n_genes) 表达矩阵
    edge_index: (2, E) 节点全局索引，0..N-1 为细胞，N..N+G-1 为基因
    N: 细胞数量
    返回: (E,) 每个边对应的表达值
    """
    u = edge_index[0]  # (E,)
    v = edge_index[1]  # (E,)

    # 条件：True 表示 u 是细胞，v 是基因；False 表示 v 是细胞，u 是基因
    mask = u < v  # (E,) 布尔张量

    # 根据条件选择细胞索引
    cell_idx = torch.where(mask, u, v)  # (E,)
    # 根据条件选择基因索引（需要减去偏移量 N）
    gene_idx = torch.where(mask, v - N, u - N)  # (E,)

    # 批量索引提取表达值
    expr_val = expr[cell_idx, gene_idx]  # (E,)
    return expr_val,cell_idx,gene_idx

# ==========================================================
# DEVICE
# ==========================================================

DEVICE=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# ==========================================================
# EMBEDDING
# ==========================================================

@torch.no_grad()

def evaluate_embeddings(

    model,

    P,

    X,

    edge_index,

    weight

):

    model.eval()

    _,_,H,_=model(

        P,

        X,

        edge_index,

        weight

    )

    return (

        H
        .cpu()
        .numpy()

    )


# ==========================================================
# MAIN
# ==========================================================

if __name__=="__main__":

    # PATH="./output_graph_dataset/S2"
    # name="a"

    PATH="./Dataset/S4"
    name = "d"

    print(
        "load"
    )

    X,\
    P,\
    Pn,\
    N,\
    G,\
    Xc,\
    Xgene,\
    L_c=load_group(
        PATH,name
    )

    X=torch.tensor(
        X
    ).float().to(
        DEVICE
    )



    P=torch.tensor(
        P
    ).float().to(
        DEVICE
    )



    # 假设 P 已经是 torch.tensor，形状 (N, F)
    mean = P.mean(dim=0, keepdim=True)  # 每个特征的均值 (1, F)
    std = P.std(dim=0, keepdim=True)  # 每个特征的标准差 (1, F)
    P_norm = (P - mean) / (std + 1e-8)  # 加小量防止除零

    # encoder = (
    #     SpatialFeatureEncoder(
    #         k=12
    #     )
    # ).cuda()
    #
    # P, \
    #     edge_space = encoder(
    #     P
    # )

    Xc=torch.tensor(
        Xc
    ).float().to(
        DEVICE
    )

    Xgene=torch.tensor(
        Xgene
    ).float().to(
        DEVICE
    )


    # ====================================
    # BUILD GRAPH
    # ====================================
    print(
        "graph"
    )
    edge_expr,\
    weight=(
        build_expression_graph(
            X
        )
    )

    # X, core_u_x, core_i_x, max_b = get_coredata(X,edge_expr,DEVICE)

    # X, core_u_x, core_i_x, max_b = get_coredata(X,edge_expr,DEVICE)
    k = 20
    hidden = 4
    lr = 5e-5
    X_use = True
    batch_size = 256
    num_neighbors = [2,2,2]
    print(f'k={k},hidden={hidden},lr={lr}')

    if X_use:
        edge_space, edge_weight=(
            build_spatial_graph(
                P,X,
                k=k
            )
        )
    else:
        edge_space, edge_weight=(
            build_spatial_graph(
                P,
                k=k
            )
        )




    np.save(f"{name}_labels.npy",L_c.reshape(-1))

    np.save(f"{name}_P.npy",P.cpu().numpy())

    # 假设原始数据已存在：X (N, d), edge_index (2, E), P (E, feat_dim) 边特征，L_c 标签
    # 转换为 PyG Data 对象

    # ================== 数据准备 ==================
    # 假设 X, P, edge_space, L_c 已存在
    # X: (N, d_x), P: (N, d_p), edge_space: (2, E)
    data = Data(
        x=torch.tensor(X, dtype=torch.float32).to(DEVICE),
        p=torch.tensor(P, dtype=torch.float32).to(DEVICE),  # P 作为节点属性
        edge_index=torch.tensor(edge_space, dtype=torch.long).to(DEVICE)
    )

    # ================== 邻居采样器 ==================

    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    # ================== 模型初始化 ==================
    model=GNN_OUR(
        #13 1e-5 k=12/400(common) a|b 8 k=15
        n_input=P.shape[1], n_clusters=len(np.unique(L_c)), n_enc=256+64, hidden=hidden, n_z=256+64, pre_ae_epoch=200,X=X

    ).to(
        DEVICE
    )


    optimizer=(
        torch.optim.Adam(

            model.parameters(),

            lr=lr,

            weight_decay=1e-5

        )
    )

    # ====================================
    # TRAIN
    # ====================================
    with torch.no_grad():
        _, _, z = model.ae(X)
    model.cluster_layer.data = kmeans(z.data, len(np.unique(L_c))).to(DEVICE)


    # ================== 辅助函数（全图前向） ==================
    def compute_global_p(model, data):
        """直接用全图计算目标分布 p"""
        model.eval()
        with torch.no_grad():
            _, _, q, _, _, _,_ = model(data.x, data.p, data.edge_index)
            p = target_distribution(q.cpu()).to(DEVICE)
        return p


    def collect_all_embeddings(model, data):
        """全图前向，得到所有节点的嵌入"""
        model.eval()
        with torch.no_grad():
            emb, _, _, _, _, _,_ = model(data.x, data.p, data.edge_index)
        return emb





    # ================== 训练循环 ==================
    metrics = []
    for epoch in range(1, 41):
        # 每 5 个 epoch 更新一次全局 p（首次必须计算）
        # if epoch == 1 or epoch % 5 == 0:
        #     p = compute_global_p(model, data)

        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)  # 自动包含 x, p, edge_index, n_id

            optimizer.zero_grad()
            x2, x_bar, q, z, z_pos, z_neg,h = model(batch.x, batch.p, batch.edge_index)
            p = target_distribution(q)
            ae_loss = F.mse_loss(x_bar, batch.x)
            # # 用 batch.n_id 将全局 p 映射到子图节点
            # p_sub = p[batch.n_id]
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
            c_loss = model.nei_con_loss(x2, z,h[0])

            loss = 0.5 * ae_loss + 0.4 * kl_loss + 0.1 * c_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"epoch={epoch}, avg_loss={avg_loss:.4f}")

        # 每 5 个 epoch 评估
        if epoch % 2 == 0:
            cell_emb = collect_all_embeddings(model, data)
            ari, nmi, ami, _ = evaluate_embeddings_cluster(
                cell_emb,
                torch.tensor(L_c.reshape(-1)).to(DEVICE),
                None,
                k=len(np.unique(L_c)),
                method='kmeans',
                use_gpu=True
            )
            print(f"Epoch {epoch}: ARI={ari:.4f}, NMI={nmi:.4f}, AMI={ami:.4f}")
            metrics.append((epoch, ari, nmi))

            np.save(f"{name}_cell_embeddings_epoch_{epoch:03d}.npy", cell_emb.cpu().numpy())
            torch.save({
                "epoch": epoch,
                "model": model.state_dict()
            }, f"./checkpoints/{name}_model_{epoch:03d}.pt")

    # 保存所有评估记录
    np.save(f"./checkpoints/{name}_eval_metrics.npy", np.array(metrics))

    metrics_arr = np.array(metrics)
    best_ari = np.max(metrics_arr[:, 1])
    best_epoch = int(metrics_arr[np.argmax(metrics_arr[:, 1]), 0])
    print(f"Best ARI: {best_ari:.4f} at epoch {best_epoch}")



    # # ====================================
    # # LOAD
    # # ====================================
    #
    # model.load_state_dict(
    #
    #     torch.load(
    #
    #         "./checkpoints/best.pt"
    #
    #     )["model"]
    #
    # )

    # ====================================
    # EVAL
    # ====================================

    # model.eval()
    # emb, _, _,_,_,_ = model( X, P, edge_space)
    # cell_embeddings=(
    #     emb[
    #         :N
    #     ]
    # )
    # gene_embeddings=(
    #     emb[
    #         N:
    #     ]
    # )
    # np.save(
    #
    #     "cell_embeddings.npy",
    #
    #     cell_embeddings.detach().cpu()
    #
    # )
    # np.save(
    #
    #     "gene_embeddings.npy",
    #
    #     gene_embeddings.detach().cpu()
    #
    # )
    # print(
    #     cell_embeddings.shape
    # )
    #
    # print(
    #     gene_embeddings.shape
    # )

    # # 假设已有 cell_embeddings (numpy), true_labels, spatial_coords (P) model.expr.cell_emb.weight.cpu().detach().numpy()
    # ari, fig = cluster_and_compare(
    #     cell_embeddings=cell_embeddings.detach().cpu(),
    #     true_labels=L_c.reshape(-1),
    #     spatial_coords=P.cpu(),
    #     method='pca',  # 快速预览，可改为 'umap' 或 'rapids'
    #     save_path='clustering_comparison.png'
    # )

    # cluster_and_compare(
    #     cell_embeddings.detach().cpu().numpy(),
    #     L_c.reshape(-1),
    #     P.cpu().numpy(),
    #     cluster_method="louvain"
    # )
