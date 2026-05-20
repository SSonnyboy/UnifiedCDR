"""
UnifiedCDR: 数据增强监督解耦的跨域推荐模型
===============================================================
1. Embedding Layer     → 双域独立的用户/物品嵌入表
2. Graph Encoder       → LightGCN 风格图卷积，独立编码两个域
3. Transfer Layer      → 按交互活跃度加权融合重叠用户
4. Disentangle Layer   → 门控投影分离 common / specific
5. Fusion Layer        → Attention 加权融合三层信息
6. Augmentation        → 局部/跨域增强
7. Multi-task Loss     → BPR + 增强 + 解耦约束 + L2
"""

import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedCDR(nn.Module):
    def __init__(self, args):
        super(UnifiedCDR, self).__init__()
        params = args.params
        self.device = args.device
        self.dtype = args.torch_type

        # ---------- 超参数 ----------
        self.emb_dim = params['embedding_size']      # 64
        self.n_layers = params['n_layers']           # 2
        self.reg_weight = params['reg_weight']       # 0.0001
        self.drop_rate = params['drop_rate']         # 0.2
        self.neg_ratio = params['neg_ratio']         # 1
        self.temperature = params['temperature']     # 0.1 not used

        # CrossAug 增强参数
        self.local_aug = params['local_alpha']       # 0.7
        self.cross_domain_aug = params['cd_alpha']   # 0.7
        self.local_lambda = params['local_lambda']   # 0.5  4loss
        self.cross_domain_lambda = params['cd_lambda']   # 0.5  4loss
        self.cd_batch = params['cbatch_size']        # 512

        # DGCDR 解耦参数
        self.cl_sim_weight = params['cl_sim_weight']  # 0.01
        self.cl_org_weight = params['cl_org_weight']  # 1.0

        # ---------- 数据维度 ----------
        self.n_shared_users = args.data['n_shared_users']   # 500
        self.d1_n_users = args.d1['n_users']                # 3000
        self.d2_n_users = args.d2['n_users']                # 2000
        self.d1_n_items = args.d1['n_items']                # 2000
        self.d2_n_items = args.d2['n_items']                # 1500
        self.n_users = self.d1_n_users + self.d2_n_users - self.n_shared_users   # 4500
        self.n_items = self.d1_n_items + self.d2_n_items                           # 3500

        # GNN concat后的输出维度
        self.gnn_dim = self.emb_dim * (self.n_layers + 1)  # 64 * 3 = 192
        self.common_dim = self.gnn_dim // 2                # 96，解耦后每半维度

        # ---------- 嵌入层 ----------
        self.d1_user_emb = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d1_item_emb = nn.Embedding(self.d1_n_items, self.emb_dim, dtype=self.dtype)
        self.d2_user_emb = nn.Embedding(self.n_users, self.emb_dim, dtype=self.dtype)
        self.d2_item_emb = nn.Embedding(self.d2_n_items, self.emb_dim, dtype=self.dtype)

        # ---------- 图结构 ----------
        self.d1_norm_adj, self.d1_user_degree = self._get_norm_adj(args.d1["inter_mat"])
        self.d2_norm_adj, self.d2_user_degree = self._get_norm_adj(args.d2["inter_mat"])

        # 跨域用户迁移的度加权
        user_laplace = self.d1_user_degree + self.d2_user_degree + 1e-7
        self.d1_user_degree = (self.d1_user_degree / user_laplace).to(self.dtype).unsqueeze(1)
        self.d2_user_degree = (self.d2_user_degree / user_laplace).to(self.dtype).unsqueeze(1)

        # ---------- 监督解耦网络 ----------
        # 投影到半维度: gnn_dim(192) → common_dim(96) + specific_dim(96)
        # 用户解耦
        self.d1_en_common = nn.Linear(self.gnn_dim, self.common_dim, dtype=self.dtype)
        self.d1_en_specific = nn.Linear(self.gnn_dim, self.common_dim, dtype=self.dtype)
        self.d2_en_common = nn.Linear(self.gnn_dim, self.common_dim, dtype=self.dtype)
        self.d2_en_specific = nn.Linear(self.gnn_dim, self.common_dim, dtype=self.dtype)
        # 物品解耦
        self.d1_item_common = nn.Linear(self.gnn_dim, self.common_dim, dtype=self.dtype)
        self.d1_item_specific = nn.Linear(self.gnn_dim, self.common_dim, dtype=self.dtype)
        self.d2_item_common = nn.Linear(self.gnn_dim, self.common_dim, dtype=self.dtype)
        self.d2_item_specific = nn.Linear(self.gnn_dim, self.common_dim, dtype=self.dtype)

        # ---------- Attention 融合层 ----------
        # 对 common/specific 各学1个权重，加权后拼接恢复 gnn_dim
        self.d1_fuse_att = nn.Linear(self.common_dim * 2, 2, dtype=self.dtype)
        self.d2_fuse_att = nn.Linear(self.common_dim * 2, 2, dtype=self.dtype)

        self.dropout = nn.Dropout(self.drop_rate)
        self.apply(self._xavier_init)

    def _xavier_init(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

    def _get_norm_adj(self, adj_mat):
        """对称归一化: D^{-1/2} A D^{-1/2}"""
        rowsum = np.array(adj_mat.sum(1)).flatten() + 1e-7
        users_degree = torch.from_numpy(rowsum[:self.n_users]).to(self.device)
        r_inv = np.power(rowsum, -0.5)
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = sp.diags(r_inv)
        adj = adj_mat.dot(r_mat_inv).transpose().dot(r_mat_inv).tocoo()
        indices = torch.LongTensor(np.array([adj.row, adj.col]))
        data = torch.tensor(adj.data, dtype=self.dtype)
        norm_adj = torch.sparse_coo_tensor(indices, data, torch.Size(adj.shape),
                                           dtype=self.dtype, device=self.device)
        return norm_adj, users_degree

    # ==================== 1. 图编码器 ====================
    def _graph_layer(self, adj_matrix, all_emb):
        side_emb = torch.sparse.mm(adj_matrix, all_emb)
        new_emb = side_emb + torch.mul(all_emb, side_emb) + all_emb
        new_emb = self.dropout(new_emb)
        return new_emb

    def _gnn_encode(self, user_emb, item_emb, norm_adj):
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        emb_list = [all_emb]
        for _ in range(self.n_layers):
            all_emb = self._graph_layer(norm_adj, all_emb)
            emb_list.append(all_emb)
        return torch.cat(emb_list, dim=-1)  # [N, gnn_dim]

    def _transfer_layer(self, d1_user_emb, d2_user_emb):
        """CrossAug: 按用户交互活跃度加权融合共享用户（仅对重叠用户）"""
        common = self.d1_user_degree * d1_user_emb + self.d2_user_degree * d2_user_emb
        d1_out = (common + d1_user_emb) / 2
        d2_out = (common + d2_user_emb) / 2
        # 非重叠用户只在单域有交互，跳过融合避免引入噪声
        d1_out[self.n_shared_users:] = d1_user_emb[self.n_shared_users:]
        d2_out[self.n_shared_users:] = d2_user_emb[self.n_shared_users:]
        return d1_out, d2_out

    # ==================== 2. 监督解耦层 ====================
    def _disentangle(self, emb, proj_common, proj_specific):
        """
        投影解耦: gnn_dim → common_dim + specific_dim (各一半)
        common + specific 维度之和 = gnn_dim，信息守恒
        """
        common = proj_common(emb)       # [N, gnn_dim] → [N, common_dim]
        specific = proj_specific(emb)   # [N, gnn_dim] → [N, common_dim]
        return common, specific

    def _fuse_attention(self, common, specific, att_layer):
        """Attention融合: 对 common/specific 各学1个权重，加权后拼接恢复原维度"""
        att = F.softmax(att_layer(torch.cat([common, specific], dim=-1)), dim=-1)
        weighted_common = att[:, 0:1] * common
        weighted_specific = att[:, 1:2] * specific
        return torch.cat([weighted_common, weighted_specific], dim=-1)  # [N, gnn_dim]

    # ==================== 3. 前向传播 ====================
    def forward(self):
        # 初始嵌入
        d1_u = self.d1_user_emb.weight
        d1_i = self.d1_item_emb.weight
        d2_u = self.d2_user_emb.weight
        d2_i = self.d2_item_emb.weight

        # 跨域迁移（共享用户先融合）
        d1_u, d2_u = self._transfer_layer(d1_u, d2_u)

        # GNN编码
        d1_all = self._gnn_encode(d1_u, d1_i, self.d1_norm_adj)
        d2_all = self._gnn_encode(d2_u, d2_i, self.d2_norm_adj)

        # 拆分用户/物品
        d1_u_gnn = d1_all[:self.n_users]
        d1_i_gnn = d1_all[self.n_users:self.n_users + self.d1_n_items]
        d2_u_gnn = d2_all[:self.n_users]
        d2_i_gnn = d2_all[self.n_users:self.n_users + self.d2_n_items]

        # --- 监督解耦: 全部用户（增强损失需要，解耦约束仍只对重叠用户计算） ---
        d1_u_c, d1_u_s = self._disentangle(d1_u_gnn,
                                           self.d1_en_common, self.d1_en_specific)
        d2_u_c, d2_u_s = self._disentangle(d2_u_gnn,
                                           self.d2_en_common, self.d2_en_specific)

        # 物品侧: 全部物品都做解耦
        d1_i_c, d1_i_s = self._disentangle(d1_i_gnn, self.d1_item_common, self.d1_item_specific)
        d2_i_c, d2_i_s = self._disentangle(d2_i_gnn, self.d2_item_common, self.d2_item_specific)

        # --- Attention融合（加权后拼接恢复原维度） ---
        d1_user_final = self._fuse_attention(d1_u_c, d1_u_s, self.d1_fuse_att)
        d2_user_final = self._fuse_attention(d2_u_c, d2_u_s, self.d2_fuse_att)

        d1_item_final = self._fuse_attention(d1_i_c, d1_i_s, self.d1_fuse_att)
        d2_item_final = self._fuse_attention(d2_i_c, d2_i_s, self.d2_fuse_att)

        # 保存解耦结果供损失计算（全部用户 + 全部物品）
        self.disentangle_info = {
            'd1_common_u': d1_u_c, 'd1_specific_u': d1_u_s,   # [4500, 192]
            'd2_common_u': d2_u_c, 'd2_specific_u': d2_u_s,   # [4500, 192]
            'd1_common_i': d1_i_c, 'd1_specific_i': d1_i_s,   # [2000, 192]
            'd2_common_i': d2_i_c, 'd2_specific_i': d2_i_s,   # [1500, 192]
        }

        return d1_user_final, d1_item_final, d2_user_final, d2_item_final

    # ==================== 4. 损失函数 ====================
    def get_score(self, u, i):
        return torch.mul(F.normalize(u, dim=-1), F.normalize(i, dim=-1)).sum(dim=-1)

    def bpr_loss(self, pos, neg):
        return -torch.mean(torch.log(torch.sigmoid(pos - neg) + 1e-24))

    def _local_augment(self, users, pos, neg, pos_c, pos_s, neg_c, neg_s):
        """
        域内增强: 用解耦层产出的 common/specific 交叉重组
        - pos_c, pos_s: 正样本的 common 和 specific（来自 _disentangle）
        - neg_c, neg_s: 负样本的 common 和 specific
        """
        c1 = torch.cat([pos_c, neg_s], dim=-1)   # 正样本共享 + 负样本特有
        c2 = torch.cat([neg_c, pos_s], dim=-1)   # 负样本共享 + 正样本特有

        c1_aug = self.local_aug * (neg - c1) + (1 - self.local_aug) * (c1 - pos)
        c2_aug = self.local_aug * (neg - c2) + (1 - self.local_aug) * (c2 - pos)

        s1 = self.get_score(users, c1_aug)
        s2 = self.get_score(users, c2_aug)
        return s1, s2

    def _cross_domain_augment(self, d1_u, d1_pos, d2_u, d2_neg, d1_u_c, d1_p_c, d2_u_c, d2_n_c):
        """
        跨域增强: 只用解耦层产出的 common 做跨域匹配
        - common 编码跨域可迁移偏好，适合做跨域对齐
        - specific 不参与跨域匹配，尊重解耦语义
        - 余弦相似度统一尺度，与BPR损失一致
        """
        pos_sc = self.get_score(d1_u, d1_pos)
        neg_sc = self.get_score(d2_u, d2_neg)

        c1 = self.get_score(d1_u_c, d2_n_c)   # Movie用户common vs Music负样本common
        c2 = self.get_score(d2_u_c, d1_p_c)   # Music用户common vs Movie正样本common

        loss = -torch.mean(torch.log(torch.sigmoid(neg_sc - c1) + 1e-24))
        loss += -torch.mean(torch.log(torch.sigmoid(c2 - pos_sc) + 1e-24))
        return loss

    def _disentangle_loss(self):
        """DGCDR解耦约束: 只对重叠用户施加 common相似 + common/specific正交"""
        info = self.disentangle_info
        # 只取重叠用户（前 n_shared_users 个）
        c1 = F.normalize(info['d1_common_u'][:self.n_shared_users], dim=1)
        c2 = F.normalize(info['d2_common_u'][:self.n_shared_users], dim=1)
        s1 = F.normalize(info['d1_specific_u'][:self.n_shared_users], dim=1)
        s2 = F.normalize(info['d2_specific_u'][:self.n_shared_users], dim=1)

        # Similarity: 两个域的common表示应该一致（跨域可迁移性）
        sim_loss = -torch.mean(torch.sum(c1 * c2, dim=1))

        # Orthogonality: common与specific应正交（分离性）平方使得值趋于0 两者正交
        ort_loss = torch.mean(torch.sum(c1 * s1, dim=1) ** 2)
        ort_loss += torch.mean(torch.sum(c2 * s2, dim=1) ** 2)

        return self.cl_sim_weight * sim_loss + self.cl_org_weight * ort_loss

    def calculate_loss(self, inter_d1, neg_d1_item, inter_d2, neg_d2_item):
        """
        联合损失
        inter_d1: [B, 3] (user, item, rating) — 训练时rating恒为1
        neg_d1_item: [B, neg_ratio]
        """
        d1_u_emb, d1_i_emb, d2_u_emb, d2_i_emb = self.forward()

        # 解耦后的物品 common/specific（供增强损失使用）
        info = self.disentangle_info
        d1_i_c, d1_i_s = info['d1_common_i'], info['d1_specific_i']
        d2_i_c, d2_i_s = info['d2_common_i'], info['d2_specific_i']

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

            # 域内增强（用解耦层产出的 common/specific 交叉重组）
            if self.local_lambda > 0:
                s1, s2 = self._local_augment(d1_u, d1_pos, d1_neg,
                                             d1_i_c[inter_d1[:, 1]], d1_i_s[inter_d1[:, 1]],
                                             d1_i_c[neg_d1_item[:, i]], d1_i_s[neg_d1_item[:, i]])
                bpr_loss_total += self.local_lambda * (
                    self.bpr_loss(self.get_score(d1_u, d1_neg), s1) +
                    self.bpr_loss(self.get_score(d1_u, d1_neg), s2)
                )
                s1, s2 = self._local_augment(d2_u, d2_pos, d2_neg,
                                             d2_i_c[inter_d2[:, 1]], d2_i_s[inter_d2[:, 1]],
                                             d2_i_c[neg_d2_item[:, i]], d2_i_s[neg_d2_item[:, i]])
                bpr_loss_total += self.local_lambda * (
                    self.bpr_loss(self.get_score(d2_u, d2_neg), s1) +
                    self.bpr_loss(self.get_score(d2_u, d2_neg), s2)
                )

            # 跨域增强（双向，用解耦层产出的 common 做跨域匹配）
            if self.cross_domain_lambda > 0:
                n = min(self.cd_batch, len(inter_d1), len(inter_d2))
                idx1 = np.random.randint(len(inter_d1), size=n)
                idx2 = np.random.randint(len(inter_d2), size=n)
                # 方向1: Movie正样本 vs Music负样本
                bpr_loss_total += self.cross_domain_lambda * self._cross_domain_augment(
                    d1_u[idx1], d1_pos[idx1], d2_u[idx2], d2_neg[idx2],
                    info['d1_common_u'][inter_d1[idx1, 0]], d1_i_c[inter_d1[idx1, 1]],
                    info['d2_common_u'][inter_d2[idx2, 0]], d2_i_c[neg_d2_item[idx2, i]]
                )
                # 方向2: Music正样本 vs Movie负样本
                bpr_loss_total += self.cross_domain_lambda * self._cross_domain_augment(
                    d2_u[idx2], d2_pos[idx2], d1_u[idx1], d1_neg[idx1],
                    info['d2_common_u'][inter_d2[idx2, 0]], d2_i_c[inter_d2[idx2, 1]],
                    info['d1_common_u'][inter_d1[idx1, 0]], d1_i_c[neg_d1_item[idx1, i]]
                )

        bpr_loss_total = bpr_loss_total / (1 + self.neg_ratio)

        # 解耦约束
        dis_loss = self._disentangle_loss()

        # L2正则
        reg = (torch.norm(d1_u) ** 2 + torch.norm(d1_pos) ** 2 +
               torch.norm(d2_u) ** 2 + torch.norm(d2_pos) ** 2) / len(d1_u)

        return bpr_loss_total + dis_loss + self.reg_weight * reg

    # ==================== 5. 评估 ====================
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
                # 统计每个样本有多少个负候选分数高于正样本 排名 = "比正样本高的数量" + 1  
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
