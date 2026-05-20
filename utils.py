"""
数据加载与预处理工具
功能：
1. 加载 train/valid/test 三元组
2. 构建稀疏邻接矩阵（用于GNN消息传播）
3. 负采样
"""
import os
import torch
import scipy.sparse as sp
import numpy as np
import random
from collections import defaultdict


def load_tri(path, mode):
    """从文本文件加载交互数据"""
    with open(os.path.join(path, f"{mode}.txt"), "r") as file:
        lines = file.readlines()
        n = int(lines[0].strip())
        data = [tri.strip('\n').split('\t') for tri in lines[1:]]
        data = [list(map(int, tri)) for tri in data]
    # train只保留正样本(rating=1)，valid/test是 [user, pos_item, neg1, neg2, ...]
    if mode == "train":
        data = [item for item in data if item[2] == 1]
    return data


#   为什么 Train 只保留正样本？
#   ───────────────────────────

#   Train 文件里本来就有正有负（来自 filter.py 中 rating < 3.5 的记录），但：

#   • 训练时的负样本是动态采样的，每轮都不一样
#   • 保留文件里的负样本意义不大，反而会让训练集变大、变慢
#   • 只保留正样本，负样本交给 construct_neg() 按需生成
# Valid/Test 保留完整的 [user, pos, neg1, neg2, ..., neg499]，用于评估。

def get_inter_mat(args):
    """
    构建三个稀疏邻接矩阵并缓存
    - d1_inter_mat: domain1 的用户-物品交互图（对称）
    - d2_inter_mat: domain2 的用户-物品交互图（对称）
    - overall_mat:  两个域合并后的全局图
    """
    d1_n_users, d2_n_users = args.d1['n_users'], args.d2['n_users']
    n_users = d1_n_users + d2_n_users - args.n_shared_users
    d1_n_items, d2_n_items = args.d1['n_items'], args.d2['n_items']
    n_items = d1_n_items + d2_n_items

    # 缓存路径
    check_paths(f"materials/{args.dataset}", f"materials/{args.dataset}/{args.domains}")
    path_root = f"materials/{args.dataset}/{args.domains}"

    # 如果缓存存在，直接加载
    if all(os.path.exists(f"{path_root}/{name}") for name in
           ["d1_inters.npy", "d2_inters.npy", "overall_inters.npy"]):
        for d, name, n_u, n_i in [(args.d1, 'd1', n_users, d1_n_items),
                                    (args.d2, 'd2', n_users, d2_n_items)]:
            inters = np.load(f"{path_root}/{name}_inters.npy")
            row, col = inters[:, 0], inters[:, 1]
            data = np.ones(len(inters))
            d['inter_mat'] = sp.coo_matrix((data, (row, col)), shape=(n_u + n_i, n_u + n_i))
        overall = np.load(f"{path_root}/overall_inters.npy")
        row, col = overall[:, 0], overall[:, 1]
        overall_data = np.ones(len(overall))
        args.overall_mat = sp.coo_matrix((overall_data, (row, col)), shape=(n_users + n_items, n_users + n_items))
        return

    # ---------- 构建 domain1 邻接矩阵 ----------
    d1_inters, d1_overall = [], []
    for inter in args.d1['train']:
        if inter[2] == 1:
            u, v = inter[0], inter[1]
            d1_inters.append((u, v + n_users))      # 物品偏移n_users
            d1_overall.append((u, v + n_users))
    d1_inters += [(v, u) for (u, v) in d1_inters]   # 对称化
    d1_overall += [(v, u) for (u, v) in d1_overall]

    d1_arr = np.array(d1_inters)
    np.save(f"{path_root}/d1_inters.npy", d1_arr)
    args.d1['inter_mat'] = sp.coo_matrix((np.ones(len(d1_arr)), (d1_arr[:, 0], d1_arr[:, 1])),
                                         shape=(n_users + d1_n_items, n_users + d1_n_items))

    # ---------- 构建 domain2 邻接矩阵 ----------
    d2_inters, d2_overall = [], []
    for inter in args.d2['train']:
        if inter[2] == 1:
            u, v = inter[0], inter[1]
            # domain2的非共享用户需要偏移
            if u < args.n_shared_users:
                u_id = u
            else:
                u_id = u + d1_n_users - args.n_shared_users
            d2_inters.append((u_id, v + n_users))
            d2_overall.append((u_id, v + n_users + d1_n_items))  # 物品在全局中偏移
    d2_inters += [(v, u) for (u, v) in d2_inters]
    d2_overall += [(v, u) for (u, v) in d2_overall]

    d2_arr = np.array(d2_inters)
    np.save(f"{path_root}/d2_inters.npy", d2_arr)
    args.d2['inter_mat'] = sp.coo_matrix((np.ones(len(d2_arr)), (d2_arr[:, 0], d2_arr[:, 1])),
                                         shape=(n_users + d2_n_items, n_users + d2_n_items))

    # ---------- 构建全局 overall 邻接矩阵 ----------
    overall = np.array(d1_overall + d2_overall)
    np.save(f"{path_root}/overall_inters.npy", overall)
    args.overall_mat = sp.coo_matrix((np.ones(len(overall)), (overall[:, 0], overall[:, 1])),
                                     shape=(n_users + n_items, n_users + n_items))


# 为什么训练不需要负交互数据？
#  GNN 的假设是：邻居节点是相似的。如果你的"邻居"里既有喜欢的又有讨厌的，模型学出来的用户表示就是"喜欢+讨厌"的平均，失去了区分度。
#   推荐系统的目标是给用户推荐他喜欢的、不给他推荐他讨厌的。如果 GNN 把讨厌的也编码进用户表示，模型就搞不清"这个用户到底喜欢什么风格"了。
#   负样本在训练里怎么用？

def construct_inter(config):  # input [[user, item, rating]...]
    """统计每个用户的交互次数和交互物品集合"""
    user2count = defaultdict(int)
    user2item = defaultdict(set)
    for u, i, r in config["train"]:
        if r == 1:
            user2count[u] += 1
            user2item[u].add(i)
    return user2count, user2item


def construct_neg(args, train_user, user2count, user2item, n_items):
    """
    为每个正样本生成负样本
    train_user: [N] 训练样本的用户ID列表（已按用户排序）
    返回: [N, neg_ratio] 负样本物品ID
    """
    neg_ratio = args.params["neg_ratio"]
    total_neg_num = len(train_user) * neg_ratio
    neg_pool = []
    cnt, user_idx, next_user_cnt = 0, -1, 0

    while cnt < total_neg_num:
        # 批量生成候选负样本，过滤掉已交互的
        batch_size = min((total_neg_num - cnt) * 2, 100000)
        neg_data = np.random.randint(0, n_items, size=batch_size)
        for item in neg_data:
            if cnt == next_user_cnt:
                user_idx += 1
                next_user_cnt += user2count[train_user[user_idx]] * neg_ratio
            if item not in user2item[train_user[user_idx]]:
                neg_pool.append(item)
                cnt += 1
            if cnt == total_neg_num:
                break

    neg_pool = np.array(neg_pool).reshape(len(train_user), neg_ratio)
    return neg_pool

# 负样本采样
#   main.py
#     │
#     ├── d1_train = [[0,10,1], [0,20,1], [0,30,1],    ← 用户0的3个正样本
#     │               [1,40,1], [1,50,1],               ← 用户1的2个正样本
#     │               [2,60,1]]                         ← 用户2的1个正样本
#     │
#     ├── d1_train[:, 0] = [0, 0, 0, 1, 1, 2]         ← train_user
#     │
#     ├── construct_inter() → user2count={0:3, 1:2, 2:1}
#     │                     → user2item={0:{10,20}, 1:{40}, 2:{60}}
#     │
#     └── construct_neg(train_user, user2count, user2item, n_items=100)
#             │
#             ├── 需要 6 个负样本
#             ├── 批量生成 [5,10,7,8,3,99,15,20,6,11,8,9]
#             ├── 过滤掉 10(用户0已交互), 20(用户0已交互)
#             └── 结果 [[5],[7],[8],[3],[99],[15]]  ← [6, 1]

#     最终训练：
#       d1_pos = [[0,10,1], [0,20,1], [0,30,1], [1,40,1], [1,50,1], [2,60,1]]
#       d1_neg = [[5], [7], [8], [3], [99], [15]]


def load_data(args, config):
    """统一数据加载入口"""
    print(f"Dataset: {args.dataset} | Domains: {args.domains}")
    args.n_shared_users = config["n_shared_users"]
    args.d1 = config["domain_1"]
    args.d2 = config["domain_2"]

    # 加载三个集合
    path_1 = os.path.join(config['path'], config['domain_1']['name'])
    args.d1["train"] = load_tri(path_1, "train")
    args.d1["valid"] = load_tri(path_1, "valid")
    args.d1["test"] = load_tri(path_1, "test")

    path_2 = os.path.join(config['path'], config['domain_2']['name'])
    args.d2["train"] = load_tri(path_2, "train")
    args.d2["valid"] = load_tri(path_2, "valid")
    args.d2["test"] = load_tri(path_2, "test")

    args.material_path = os.path.join("./materials", args.dataset, args.domains)
    get_inter_mat(args)
    args.data = config
    args.torch_type = torch.float32  # 用float32更省显存


def set_device(gpu_id):
    if gpu_id == -1:
        return torch.device('cpu')
    return torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def check_paths(*paths):
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
