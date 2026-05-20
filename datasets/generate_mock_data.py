"""
Mock 数据生成器
===============
生成符合真实 Amazon Movie-Music 预处理流程后的数据格式，
字段、文件结构、ID 映射逻辑与真实数据完全一致。

输出格式（与 process.py 输出一致）：
- user2index.txt:  第一行 "total_users overlap"，之后 "new_id\traw_id"
- item2index.txt:  第一行 "total_items"，之后 "new_id\traw_id"
- train.txt:       第一行 "n_train"，之后 "user\titem\trating"
- valid.txt/test.txt: 第一行 "n_valid"，之后 "user\tpos_item\tneg1\t...\tneg499"

ID 映射规则（与真实数据一致）：
- Domain1(Movie): 用户新 ID 0~2999，共享用户 0~499
- Domain2(Music): 用户新 ID 0~1999，共享用户 0~499，非共享用户 500~1999
"""
import os
import random
import argparse


def generate_domain(name, n_users, n_items, n_interactions, n_eval_per_user,
                    n_shared, is_domain2=False, d1_n_users=0, seed=42):
    """为单个域生成交互数据"""
    random.seed(seed)

    # ========== 原始 ID 设计 ==========
    # Domain1: 原始 ID 0 ~ n_users-1
    # Domain2: 共享用户原始 ID 0 ~ n_shared-1
    #          非共享用户原始 ID d1_n_users ~ d1_n_users + n_users - n_shared - 1
    # 这样两个域的原始 ID 不会冲突（与真实 Amazon 不同域用户 ID 不冲突一致）
    if is_domain2:
        shared_users = list(range(n_shared))
        non_shared_users = list(range(d1_n_users, d1_n_users + n_users - n_shared))
        raw_users = shared_users + non_shared_users
        # 映射后的新 ID：共享 0~n_shared-1，非共享 n_shared~n_users-1
        raw2new = {u: i for i, u in enumerate(raw_users)}
    else:
        raw_users = list(range(n_users))
        raw2new = {u: u for u in raw_users}

    items = list(range(n_items))

    # ========== 生成训练交互（保证 5-core） ==========
    train_inters = []
    user_items = {raw2new[u]: set() for u in raw_users}

    # 每个用户至少 5 个交互
    for u_raw in raw_users:
        u = raw2new[u_raw]
        for _ in range(5):
            while True:
                item = random.choice(items)
                if item not in user_items[u]:
                    user_items[u].add(item)
                    train_inters.append((u, item, 1))
                    break

    # 随机填充到目标交互数
    all_user_ids = [raw2new[u] for u in raw_users]
    while len(train_inters) < n_interactions:
        u = random.choice(all_user_ids)
        item = random.choice(items)
        if item not in user_items[u]:
            user_items[u].add(item)
            train_inters.append((u, item, 1))

    random.shuffle(train_inters)

    # ========== 生成验证集和测试集 ==========
    valid_data = []
    test_data = []

    for u_raw in raw_users:
        u = raw2new[u_raw]
        pos_items = list(user_items[u])
        if len(pos_items) < 2:
            continue

        # 取最后两个正样本作为 valid/test 正样本
        v_pos = pos_items[-2]
        t_pos = pos_items[-1]

        # 各配 499 个负样本
        all_negs = [i for i in items if i not in user_items[u]]
        if len(all_negs) < 499:
            all_negs = all_negs * (499 // len(all_negs) + 1)

        v_negs = random.sample(all_negs, 499)
        t_negs = random.sample(all_negs, 499)

        valid_data.append([u, v_pos] + v_negs)
        test_data.append([u, t_pos] + t_negs)

    return train_inters, valid_data, test_data, raw_users, items, raw2new


def save_domain(path, name, train, valid, test, raw_users, items, raw2new, overlap):
    """保存单个域的数据文件，格式与 process.py 输出完全一致"""
    os.makedirs(path, exist_ok=True)

    # ---------- user2index.txt ----------
    # 第一行: "total_users overlap"
    # 之后每行: "new_id\traw_id"
    # 顺序: 共享用户在前，非共享用户在后（与 process.py 一致）
    with open(os.path.join(path, "user2index.txt"), "w") as f:
        f.write(f"{len(raw_users)} {overlap}\n")
        for new_id, raw_id in enumerate(raw_users):
            f.write(f"{new_id}\t{raw_id}\n")

    # ---------- item2index.txt ----------
    with open(os.path.join(path, "item2index.txt"), "w") as f:
        f.write(f"{len(items)}\n")
        for idx, i in enumerate(items):
            f.write(f"{idx}\t{i}\n")

    # ---------- train.txt ----------
    with open(os.path.join(path, "train.txt"), "w") as f:
        f.write(f"{len(train)}\n")
        for u, i, r in train:
            f.write(f"{u}\t{i}\t{r}\n")

    # ---------- valid.txt ----------
    with open(os.path.join(path, "valid.txt"), "w") as f:
        f.write(f"{len(valid)}\n")
        for row in valid:
            f.write("\t".join(map(str, row)) + "\n")

    # ---------- test.txt ----------
    with open(os.path.join(path, "test.txt"), "w") as f:
        f.write(f"{len(test)}\n")
        for row in test:
            f.write("\t".join(map(str, row)) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='./datasets/processed/Amazon/Movie-Music', type=str)
    parser.add_argument('--n_shared_users', default=500, type=int, help='重叠用户数')
    parser.add_argument('--d1_users', default=3000, type=int, help='Movie域总用户数')
    parser.add_argument('--d1_items', default=2000, type=int, help='Movie域物品数')
    parser.add_argument('--d2_users', default=2000, type=int, help='Music域总用户数')
    parser.add_argument('--d2_items', default=1500, type=int, help='Music域物品数')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    # ========== Domain1: Movie ==========
    print("Generating Movie domain...")
    d1_train, d1_valid, d1_test, d1_raw_users, d1_items, d1_raw2new = generate_domain(
        "Movie", args.d1_users, args.d1_items,
        n_interactions=25000, n_eval_per_user=500,
        n_shared=args.n_shared_users, is_domain2=False, seed=args.seed
    )

    # ========== Domain2: Music ==========
    # Domain2 非共享用户的原始 ID 从 d1_users 开始，避免与 Domain1 冲突
    print("Generating Music domain...")
    d2_train, d2_valid, d2_test, d2_raw_users, d2_items, d2_raw2new = generate_domain(
        "Music", args.d2_users, args.d2_items,
        n_interactions=15000, n_eval_per_user=500,
        n_shared=args.n_shared_users, is_domain2=True,
        d1_n_users=args.d1_users, seed=args.seed + 1
    )

    # ========== 保存 ==========
    d1_path = os.path.join(args.output, "Movie")
    d2_path = os.path.join(args.output, "Music")

    save_domain(d1_path, "Movie", d1_train, d1_valid, d1_test,
                d1_raw_users, d1_items, d1_raw2new, overlap=args.n_shared_users)
    save_domain(d2_path, "Music", d2_train, d2_valid, d2_test,
                d2_raw_users, d2_items, d2_raw2new, overlap=args.n_shared_users)

    # ========== 统计信息 ==========
    print("\n===== Mock Data Generated =====")
    print(f"Output path: {args.output}")
    print(f"Shared users: {args.n_shared_users}")
    print(f"Movie: {args.d1_users} users, {args.d1_items} items, {len(d1_train)} train, {len(d1_valid)} valid")
    print(f"Music: {args.d2_users} users, {args.d2_items} items, {len(d2_train)} train, {len(d2_valid)} valid")
    print("\n请确认 config.yaml 中的统计数字与上述一致！")


if __name__ == '__main__':
    main()
