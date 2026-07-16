# =====================================
# trainer/pretrain.py
# =====================================

import torch
from torch_geometric.data import Dataset, DataLoader
import torch.nn.functional as F

# ============================
# 1. 构建能够按边迭代的 Dataset
# ============================
class EdgeDataset(Dataset):
    def __init__(self, edge_index, weight, expr):
        """
        edge_index: (2, E)  细胞-基因边（基因索引已偏移？此处假设未偏移，即基因用0..G-1）
        weight: (E,)        边权重
        expr: (n_cells, n_genes) 原始表达矩阵，用于取出真实表达值
        """
        self.edge_index = edge_index
        self.weight = weight
        self.expr = expr
        self.num_edges = edge_index.size(1)

    def __len__(self):
        return self.num_edges

    def __getitem__(self, idx):
        u = self.edge_index[0, idx]   # 细胞索引
        v = self.edge_index[1, idx]   # 基因索引
        w = self.weight[idx]
        # 从原始表达矩阵中取出该条边的真实表达值（用于监督）
        expr_val = self.expr[u, v]
        return {
            'cell_id': u.unsqueeze(0),
            'gene_id': v.unsqueeze(0),
            'edge': torch.stack([u, v]).unsqueeze(1),  # (2,1) 单条边
            'weight': w.unsqueeze(0),
            'expr': expr_val.unsqueeze(0)
        }

def collate_fn(batch):
    """将多条边拼接为一个 batch"""
    cell_ids = torch.cat([b['cell_id'] for b in batch], dim=0)
    gene_ids = torch.cat([b['gene_id'] for b in batch], dim=0)
    edges = torch.cat([b['edge'] for b in batch], dim=1)    # (2, B)
    weights = torch.cat([b['weight'] for b in batch], dim=0) # (B,)
    exprs = torch.cat([b['expr'] for b in batch], dim=0)    # (B,)
    return edges, weights, exprs, cell_ids, gene_ids


# ============================
# 2. 包装为满足原 pretrain 需要的 Loader
# ============================
class BatchWrapper:
    """将 batch 数据包装为具有 edge_index, weight, expr, cell_id, gene_id 属性的对象"""
    def __init__(self, edges, weights, exprs, cids, gids):
        self.edge_index = edges
        self.weight = weights
        self.expr = exprs
        self.cell_id = cids
        self.gene_id = gids

def create_loader(edge_index, weight, expr, batch_size, shuffle=True):
    dataset = EdgeDataset(edge_index, weight, expr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    for batch in dataloader:
        edges, weights, exprs, cids, gids = batch
        yield BatchWrapper(edges, weights, exprs, cids, gids)


def pretrain_loss(pred_expr, pred_edge, expr):
    """
    pred_expr: (B,) 预测的表达值（对数尺度）
    pred_edge: (B,) 预测的边存在概率
    expr: (B,) 真实的表达值（原始计数）
    """
    target_edge = (expr > 0).float()   # 真实边标签：表达值>0为1

    # 表达量重建损失：使用 log1p 稳定数值，MSE 训练平滑
    l1 = F.mse_loss(pred_expr, torch.log1p(expr))

    # 边存在二分类损失：预测边概率与真实标签交叉熵
    l2 = F.binary_cross_entropy(pred_edge, target_edge)

    loss = l1 + l2
    return {"loss": loss}

def pretrain(

    model,

    loader,

    optimizer,

    criterion,

    epochs,

    device,

    save_path

):

    model.train()

    best=1e9

    for ep in range(
        epochs
    ):

        total=0

        for batch in loader:

            edge=batch.edge_index.to(
                device
            )

            weight=batch.weight.to(
                device
            )

            expr=batch.expr.to(
                device
            )

            cid=batch.cell_id.to(
                device
            )

            gid=batch.gene_id.to(
                device
            )

            emb=model(

                edge,

                weight

            )

            pred_expr,\
            pred_edge=(
                model.decode(

                    cid,

                    gid,

                    emb

                )
            )

            loss=criterion(

                pred_expr,

                pred_edge,

                expr

            )

            optimizer.zero_grad()

            loss[
                "loss"
            ].backward()

            optimizer.step()

            total+=(
                loss[
                    "loss"
                ].item()
            )

        total/=len(
            loader
        )

        print(
            ep,
            total
        )

        if total<best:

            best=total

            torch.save({

                "model":
                model.state_dict(),

                "cell":
                model.cell_emb.weight,

                "gene":
                model.gene_emb.weight

            },

            save_path)

    return best