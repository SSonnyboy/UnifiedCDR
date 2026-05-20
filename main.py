"""
UnifiedCDR 训练入口
流程：
1. 解析参数 & 加载配置
2. 加载数据 & 构建邻接矩阵
3. 初始化模型 & 优化器
4. 训练循环（每5轮验证一次）
5. 最终测试
"""
import argparse
import sys
import yaml
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from models import UnifiedCDR, LightGCN
from utils import load_data, set_device, set_seed, construct_inter, construct_neg

MODEL_MAP = {
    'UnifiedCDR': UnifiedCDR,
    'LightGCN': LightGCN,
}


def init():
    parser = argparse.ArgumentParser(description="UnifiedCDR: 融合数据增强与监督解耦的跨域推荐")
    parser.add_argument('--model', default='UnifiedCDR', type=str,
                        choices=list(MODEL_MAP.keys()), help='模型选择')
    parser.add_argument('--domains', default='Movie-Music', type=str,
                        help='跨域数据集对，默认 Movie-Music')
    parser.add_argument('--gpu', default=0, type=int, help='GPU编号，-1表示CPU')
    parser.add_argument('--seed', default=2024, type=int, help='随机种子')
    parser.add_argument('--neg_valid_num', default=499, type=int, help='验证/测试时每个用户的负样本数')
    parser.add_argument('--epoch', default=None, type=int, help='覆盖配置中的训练轮数')
    parser.add_argument('--lr', default=None, type=float, help='覆盖配置中的学习率')
    args = parser.parse_args()

    args.dataset = "Amazon"
    args.approach = args.model

    # 设备 & 种子
    args.device = set_device(args.gpu)
    set_seed(args.seed)
    print(f"Using device: {args.device}")

    # 加载yaml配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 命令行可覆盖epoch和lr
    if args.epoch:
        config[args.approach]['epoch'] = args.epoch
    if args.lr:
        config[args.approach]['lr'] = args.lr

    # 加载数据
    load_data(args, config["datasets"][args.dataset][args.domains])

    # 加载模型
    args.params = config[args.approach]
    args.lr = args.params.get('lr', 0.001)
    args.model = MODEL_MAP[args.approach](args).to(args.device)
    args.optim = torch.optim.Adam(args.model.parameters(), lr=args.lr)

    print(f"Model: {args.approach}")
    print(f"Hyper-params: {args.params}")
    total_params = sum(p.numel() for p in args.model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    return args


def train(args):
    """主训练循环"""
    evaluate(args, epoch=0, mode="valid")

    print("\nConstructing interaction dicts for negative sampling...")
    d1_user2count, d1_user2item = construct_inter(args.d1)
    d2_user2count, d2_user2item = construct_inter(args.d2)

    # 将训练数据转为numpy并按用户排序（方便负采样）
    d1_train = np.array(args.d1["train"])
    d1_train = d1_train[np.argsort(d1_train[:, 0])]
    d2_train = np.array(args.d2["train"])
    d2_train = d2_train[np.argsort(d2_train[:, 0])]

    total_len = len(d1_train) + len(d2_train)
    batch_size = args.params["batch_size"]
    n_epoch = args.params["epoch"]

    for epoch in range(n_epoch):
        args.model.train()
        print(f"\n=== TRAIN Epoch {epoch + 1}/{n_epoch} ===")

        # 每轮重新生成负样本（动态负采样，增强泛化）
        d1_neg = construct_neg(args, d1_train[:, 0], d1_user2count, d1_user2item, args.d1["n_items"])
        d2_neg = construct_neg(args, d2_train[:, 0], d2_user2count, d2_user2item, args.d2["n_items"])

        # shuffle
        d1_idx = np.random.permutation(len(d1_train))
        d2_idx = np.random.permutation(len(d2_train))
        d1_pos, d1_neg = d1_train[d1_idx], d1_neg[d1_idx]
        d2_pos, d2_neg = d2_train[d2_idx], d2_neg[d2_idx]

        # 混合双域样本训练
        index_list = list(range(total_len))
        np.random.shuffle(index_list)

        d1_point, d2_point = 0, 0
        epoch_loss = 0
        n_batches = 0

        for batch_start in range(0, total_len, batch_size):
            indices = np.array(index_list[batch_start: min(batch_start + batch_size, total_len)])
            d1_num = np.sum(indices < len(d1_train))
            d2_num = len(indices) - d1_num

            d1_inter = torch.from_numpy(d1_pos[d1_point:d1_point + d1_num]).to(args.device)
            d1_neg_item = torch.from_numpy(d1_neg[d1_point:d1_point + d1_num]).to(args.device)
            d2_inter = torch.from_numpy(d2_pos[d2_point:d2_point + d2_num]).to(args.device)
            d2_neg_item = torch.from_numpy(d2_neg[d2_point:d2_point + d2_num]).to(args.device)

            args.optim.zero_grad()
            loss = args.model.calculate_loss(d1_inter, d1_neg_item, d2_inter, d2_neg_item)
            loss.backward()
            args.optim.step()

            epoch_loss += loss.item()
            n_batches += 1

            if n_batches % 50 == 0 or batch_start == 0:
                print(f"  Batch {n_batches:4d} | Loss: {loss.item():.4f}")

            d1_point += d1_num
            d2_point += d2_num

        print(f"Epoch {epoch + 1} average loss: {epoch_loss / n_batches:.4f}")

        # 验证
        if (epoch + 1) % 5 == 0:
            evaluate(args, epoch=epoch + 1, mode="valid")

    print("\nTraining finished!")


def evaluate(args, epoch=None, mode="test", eval_size=32):
    """评估：分别评估两个域"""
    args.model.eval()
    prefix = f"VALID Epoch {epoch}" if mode == "valid" else "TEST"
    print(f"\n--- {prefix} ---")

    eval_set_1 = DataLoader(
        TensorDataset(torch.tensor(args.d1[mode]).to(args.device)),
        batch_size=eval_size
    )
    eval_set_2 = DataLoader(
        TensorDataset(torch.tensor(args.d2[mode]).to(args.device)),
        batch_size=eval_size
    )

    h1, n1, h2, n2 = args.model.predict(eval_set_1, eval_set_2, args.neg_valid_num)
    return h1, n1, h2, n2


if __name__ == '__main__':
    args = init()
    train(args)
    print("\n" + "=" * 50)
    print("Final Test Evaluation:")
    print("=" * 50)
    evaluate(args)
