import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialFeatureEncoder(nn.Module):
    def __init__(self, k=16, use_orientation=True, normalize=True):
        super().__init__()
        self.k = k
        self.use_orientation = use_orientation
        self.normalize = normalize

    @torch.no_grad()
    def build_knn(self, P):
        N = P.shape[0]
        idx = torch.cdist(P, P).topk(self.k + 1, largest=False).indices[:, 1:]
        src = torch.arange(N, device=P.device).unsqueeze(1).repeat(1, self.k)
        edge_index = torch.stack([src.reshape(-1), idx.reshape(-1)])
        return idx, edge_index

    def polar(self, P):
        d = P - P.mean(0, keepdim=True)
        return torch.norm(d, dim=-1, keepdim=True), torch.atan2(d[:, 1], d[:, 0]).unsqueeze(-1)

    def relative(self, r, theta, idx):
        ri, rj = r[idx], r.unsqueeze(1)
        ti, tj = theta[idx], theta.unsqueeze(1)
        dt = ti - tj
        return (ri - rj).mean(1), torch.sin(dt).mean(1), torch.cos(dt).mean(1)

    def orientation(self, P, idx):
        neigh = P[idx] - P[idx].mean(1, keepdim=True)
        cov = torch.matmul(neigh.transpose(1, 2), neigh) / self.k
        return torch.linalg.eigh(cov).eigenvectors[:, :, -1]

    def forward(self, P):
        idx, edge_index = self.build_knn(P)
        r, theta = self.polar(P)
        dr, sin, cos = self.relative(r, theta, idx)
        feats = [r, dr, sin, cos]
        if self.use_orientation:
            feats.append(self.orientation(P, idx))
        S = torch.cat(feats, -1)
        if self.normalize:
            S = F.layer_norm(S, [S.shape[-1]])
        return S, edge_index