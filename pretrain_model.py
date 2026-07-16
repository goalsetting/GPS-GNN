# =====================================
# trainer/pretrain.py
# =====================================

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ============================
# 1. 构建能够按边迭代的 Dataset
# ============================
class EdgeDataset(Dataset):
    def __init__(self, edge_index, weight, expr, n_cell):
        """
        edge_index: (2, E)  细胞-基因边（基因索引已偏移？此处假设未偏移，即基因用0..G-1）
        weight: (E,)        边权重
        expr: (n_cells, n_genes) 原始表达矩阵，用于取出真实表达值
        """
        self.edge_index = edge_index
        self.weight = weight
        self.expr = expr
        self.n_cell = n_cell
        self.num_edges = edge_index.size(1)

    def __len__(self):
        return self.num_edges

    def __getitem__(self, idx):
        u = self.edge_index[0, idx]   # 细胞索引
        v = self.edge_index[1, idx]   # 基因索引
        w = self.weight[idx]
        # 提取表达值：判断哪一个是细胞，哪一个是基因
        if u < v:          # u 是细胞，v 是基因（全局索引）
            cell_idx = u
            gene_idx = v - self.n_cell
        else:                        # u 是基因，v 是细胞
            cell_idx = v
            gene_idx = u - self.n_cell

        expr_val = self.expr[cell_idx, gene_idx]   # 标量值

        return {
            'edge': torch.stack([u, v]),            # (2,)
            'weight': w,                            # 标量
            'expr': expr_val,                       # 标量
            'cell_id': cell_idx,                    # 标量
            'gene_id': gene_idx                     # 基因原始索引
        }

# def collate_fn(batch):
#     edges = torch.stack([b['edge'] for b in batch], dim=1)   # (2, B)
#     weights = torch.stack([b['weight'] for b in batch])      # (B,)
#     exprs = torch.stack([b['expr'] for b in batch])          # (B,)
#     cell_ids = torch.stack([b['cell_id'] for b in batch])    # (B,)
#     gene_ids = torch.stack([b['gene_id'] for b in batch])    # (B,)
#     return edges, weights, exprs, cell_ids, gene_ids


class BatchWrapper:
    def __init__(self, edges, weights, exprs, cids, gids):
        self.edge_index = edges
        self.weight = weights
        self.expr = exprs
        self.cell_id = cids
        self.gene_id = gids

def create_loader(edge_index, weight, expr, batch_size, n_cell, shuffle=True):
    dataset = EdgeDataset(edge_index, weight, expr, n_cell)

    def collate_fn(batch):
        edges = torch.stack([item['edge'] for item in batch], dim=1)   # (2, B)
        weights = torch.stack([item['weight'] for item in batch])      # (B,)
        exprs = torch.stack([item['expr'] for item in batch])          # (B,)
        cell_ids = torch.stack([item['cell_id'] for item in batch])    # (B,)
        gene_ids = torch.stack([item['gene_id'] for item in batch])    # (B,)
        return edges, weights, exprs, cell_ids, gene_ids

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader   # 直接返回 DataLoader 实例，而不是生成器

def pretrain_loss(pred_expr, pred_edge, expr):
    """
    pred_expr: (B,) 预测的表达值（对数尺度）
    pred_edge: (B,) 预测的边存在概率
    expr: (B,) 真实的表达值（原始计数）
    """
    target_edge = (expr > 0).float()   # 真实边标签：表达值>0为1

    # -----------------
    # only positive
    # -----------------

    mask = (expr > 0)

    if mask.sum() > 0:
        l1 = F.mse_loss(
            pred_expr[
                mask
            ],
            expr[
                mask
            ]
        )

    else:
        l1 = torch.tensor(
            0,
            device=expr.device,
            dtype=torch.float
        )

    # L2
    pos = expr > 0
    neg = expr == 0

    idx_neg = torch.where(
        neg
    )[0]

    idx_neg = idx_neg[
        torch.randperm(
            len(idx_neg)
        )[:3 * pos.sum()]
    ]
    mask = torch.zeros_like(
        expr
    ).bool()

    mask[pos] = True
    mask[idx_neg] = True

    l2 = F.binary_cross_entropy(
        pred_edge[
            mask
        ],
        target_edge[
            mask
        ]
    )

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
    # 提前将所有批次数据加载到内存
    batches = list(loader)  # 每个元素是 (edges, weights, exprs, cids, gids)
    model.train()

    best=1e9

    for ep in range(
        epochs
    ):

        total=0
        lens = 1
        for edges, weights, exprs, cids, gids in batches:  # 每个 epoch 都会重新创建迭代器
            lens += 1
            edge=edges.to(
                device
            )

            weight=weights.to(
                device
            )

            expr=exprs.to(
                device
            )

            cid=cids.to(
                device
            )

            gid=gids.to(
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


        total/=lens

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