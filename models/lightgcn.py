"""
LightGCN 基线模型 (He et al. 2020)
单域独立训练，无跨域迁移、无解耦、无增强
用于对比 UnifiedCDR 的跨域增益
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self, args):
        super(LightGCN, self).__init__()
        params = args.params
        self.device = args.device
        self.dtype = args.torch_type

        # ---------- 超参数 ----------
        self.emb_dim = params['embedding_size']      # 64
        self.n_layers = params['n_layers']           # 2
        self.reg_weight = params['reg_weight']       # 0.0001
        self.neg_ratio = params.get('neg_ratio', 1)

        # ---------- 数据维度 ----------
        self.n_shared_users = args.data['n_shared_users']
        self.d1_n_users = args.d1['n_users']
        self.d2_n_users = args.d2['n_users']
        self.d1_n_items = args.d1['n_items']
        self.d2_n_items = args.d2['n_items']
        self.n_users = self.d1_n_users + self.d2_n_users - self.n_shared_users

        # ---------- 嵌入层 ----------
        self.d1_user_emb = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d1_item_emb = nn.Embedding(self.d1_n_items, self.emb_dim, dtype=self.dtype)
        self.d2_user_emb = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d2_item_emb = nn.Embedding(self.d2_n_items, self.emb_dim, dtype=self.dtype)

        # ---------- 图结构 ----------
        self.d1_norm_adj = self._get_norm_adj(args.d1["inter_mat"])
        self.d2_norm_adj = self._get_norm_adj(args.d2["inter_mat"])

        self.apply(self._xavier_init)

    def _xavier_init(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)

    def _get_norm_adj(self, adj_mat):
        """对称归一化: D^{-1/2} A D^{-1/2}"""
        rowsum = np.array(adj_mat.sum(1)).flatten() + 1e-7
        r_inv = np.power(rowsum, -0.5)
        r_inv[np.isinf(r_inv)] = 0
        import scipy.sparse as sp
        r_mat_inv = sp.diags(r_inv)
        adj = adj_mat.dot(r_mat_inv).transpose().dot(r_mat_inv).tocoo()
        indices = torch.LongTensor(np.array([adj.row, adj.col]))
        data = torch.tensor(adj.data, dtype=self.dtype)
        return torch.sparse_coo_tensor(indices, data, torch.Size(adj.shape),
                                       dtype=self.dtype, device=self.device)

    # ==================== 图编码器 ====================
    def _gnn_encode(self, user_emb, item_emb, norm_adj):
        """标准 LightGCN: 线性传播 + 层间平均"""
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        emb_list = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            emb_list.append(all_emb)
        # 标准 LightGCN: 各层取平均（非 concat）
        return torch.stack(emb_list, dim=1).mean(dim=1)  # [N, emb_dim]

    # ==================== 前向传播 ====================
    def forward(self):
        d1_user_final = self._gnn_encode(self.d1_user_emb.weight,
                                          self.d1_item_emb.weight, self.d1_norm_adj)
        d2_user_final = self._gnn_encode(self.d2_user_emb.weight,
                                          self.d2_item_emb.weight, self.d2_norm_adj)

        d1_item_final = d1_user_final[self.n_users:self.n_users + self.d1_n_items]
        d1_user_final = d1_user_final[:self.n_users]

        d2_item_final = d2_user_final[self.n_users:self.n_users + self.d2_n_items]
        d2_user_final = d2_user_final[:self.n_users]

        return d1_user_final, d1_item_final, d2_user_final, d2_item_final

    # ==================== 损失函数 ====================
    def get_score(self, u, i):
        return torch.mul(F.normalize(u, dim=-1), F.normalize(i, dim=-1)).sum(dim=-1)

    def bpr_loss(self, pos, neg):
        return -torch.mean(torch.log(torch.sigmoid(pos - neg) + 1e-24))

    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        d1_u_emb, d1_i_emb, d2_u_emb, d2_i_emb = self.forward()

        d1_u = d1_u_emb[inter_d1[:, 0]]
        d1_pos = d1_i_emb[inter_d1[:, 1]]
        d2_u = d2_u_emb[inter_d2[:, 0]]
        d2_pos = d2_i_emb[inter_d2[:, 1]]

        bpr_loss_total = 0.0
        for i in range(self.neg_ratio):
            d1_neg = d1_i_emb[neg_d1_item[:, i]]
            d2_neg = d2_i_emb[neg_d2_item[:, i]]
            bpr_loss_total += self.bpr_loss(self.get_score(d1_u, d1_pos), self.get_score(d1_u, d1_neg))
            bpr_loss_total += self.bpr_loss(self.get_score(d2_u, d2_pos), self.get_score(d2_u, d2_neg))
        bpr_loss_total = bpr_loss_total / (1 + self.neg_ratio)

        # L2正则（对初始嵌入）
        reg = (torch.norm(d1_u_emb[inter_d1[:, 0]]) ** 2 +
               torch.norm(d1_i_emb[inter_d1[:, 1]]) ** 2 +
               torch.norm(d2_u_emb[inter_d2[:, 0]]) ** 2 +
               torch.norm(d2_i_emb[inter_d2[:, 1]]) ** 2) / len(d1_u)

        return bpr_loss_total + self.reg_weight * reg

    # ==================== 评估 ====================
    @torch.no_grad()
    def predict(self, eval_set_1, eval_set_2, neg_valid_num):
        d1_u, d1_i, d2_u, d2_i = self.forward()

        def _eval(eval_set, u_emb, i_emb, name):
            hit, ndcg, total = 0, 0, 0
            for batch in eval_set:
                b = batch[0]
                users = b[:, :1].repeat(1, 1 + neg_valid_num)
                items = b[:, 1:]
                scores = self.get_score(u_emb[users], i_emb[items])
                ranks = torch.sum(scores > scores[:, :1], dim=-1) + 1
                for rank in ranks:
                    r = rank.item()
                    if r <= 10:
                        hit += 1
                        ndcg += 1.0 / math.log2(r + 1)
                    total += 1
            h, n = (hit / total, ndcg / total) if total else (0, 0)
            print(f"{name}  Hit@10: {h:.4f}, NDCG@10: {n:.4f}")
            return h, n

        h1, n1 = _eval(eval_set_1, d1_u, d1_i, "Domain1(Movie)")
        h2, n2 = _eval(eval_set_2, d2_u, d2_i, "Domain2(Music)")
        return h1, n1, h2, n2
