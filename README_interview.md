# UnifiedCDR: 双目标跨域推荐系统

> **面试项目定位**：独立设计并实现了一个跨域推荐系统，解决单域数据稀疏和跨域负迁移问题。在 Amazon Movie-Music 场景上验证，核心模型约 300 行，技术方案可扩展至工业推荐场景。

---

## 一、项目背景与问题定义

### 1.1 业务场景
用户在多个内容平台都有行为，例如：
- 在 **Movie** 平台看过《星际穿越》《盗梦空间》
- 在 **Music** 平台听过 Hans Zimmer、Pink Floyd

但每个平台单独的数据都很稀疏（长尾用户只有几条交互），传统单域推荐模型学不好。如果直接把 Movie 的知识搬到 Music，又可能**负迁移**——推荐一堆用户根本不感兴趣的跨界内容。

### 1.2 核心挑战

| 挑战 | 说明 |
|------|------|
| **数据稀疏** | 单域交互少，模型过拟合 |
| **负迁移** | 不加区分地迁移知识，反而损害目标域 |
| **兴趣混杂** | 用户的兴趣里，哪些是跨域通用的？哪些是只在某个域才有的？ |

### 1.3 我的解决思路

> 我把问题拆解成两个子问题：**数据不够** 和 **知识没分清楚**。
>
> - 针对"数据不够"→ 设计**跨域数据增强**，在数据层面生成更多训练信号
> - 针对"知识没分清楚"→ 设计**监督解耦机制**，在表示层面显式分离"通用兴趣"和"领域特有兴趣"
> - 最后用一个统一的框架联合优化，让两者互相增强

---

## 二、模型架构

```
Input:  Domain1(Movie)          Domain2(Music)
       ┌─────────────┐         ┌─────────────┐
       │ Embedding   │         │ Embedding   │
       │  u1, i1     │         │  u2, i2     │
       └──────┬──────┘         └──────┬──────┘
              ▼                       ▼
       ┌─────────────┐         ┌─────────────┐
       │ GNN Encoder │         │ GNN Encoder │  (LightGCN × 2)
       │  n_layers   │         │  n_layers   │
       └──────┬──────┘         └──────┬──────┘
              ▼                       ▼
       ┌─────────────┐         ┌─────────────┐
       │Transfer Layer│        │Transfer Layer│
       │按交互活跃度  │         │融合重叠用户  │
       │加权融合      │         │表示          │
       └──────┬──────┘         └──────┬──────┘
              ▼                       ▼
       ┌─────────────────────────────────────┐
       │     Disentangle Layer (投影门控)    │
       │  common  = emb ⊙ sigmoid(MLP(emb))  │
       │ specific = emb ⊙ sigmoid(MLP(emb))  │
       └─────────────────────────────────────┘
              ▼                       ▼
       ┌─────────────────────────────────────┐
       │      Fusion Layer (Attention)       │
       │  fused = w_gnn·gnn + w_c·common     │
       │          + w_s·specific             │
       └─────────────────────────────────────┘
              ▼                       ▼
       ┌─────────────────────────────────────┐
       │   Augmentation (Local + Cross)      │
       │  局部: pos/neg 维度重组             │
       │  跨域: d1用户 + d2负样本            │
       └─────────────────────────────────────┘
              ▼                       ▼
       ┌─────────────────────────────────────┐
       │         Multi-Task Loss             │
       │  • BPR Loss (双域排序)              │
       │  • Local Augmentation Loss          │
       │  • Cross-Domain Augmentation Loss   │
       │  • Similarity Loss (common一致)     │
       │  • Orthogonality Loss (common⊥sp)  │
       │  • L2 Regularization                │
       └─────────────────────────────────────┘
              ▼                       ▼
         Hit@10 / NDCG@10      Hit@10 / NDCG@10
```

---

## 三、核心模块详解

### 3.1 双域 GNN 编码器

两个域各自维护独立的用户/物品嵌入表，通过 **LightGCN** 风格的消息传播进行编码：

$$H^{(l+1)} = \tilde{A} H^{(l)}, \quad \tilde{A} = D^{-1/2} A D^{-1/2}$$

所有层的输出 concat 起来，得到每个节点的最终 GNN 表示。

### 3.2 跨域用户迁移（Transfer Layer）

重叠用户在两个域都有行为，他们的表示应该如何融合？

我的做法：**按交互活跃度加权**。如果某个用户在 Movie 域交互很多、在 Music 域交互很少，那 Movie 域的表示应该占更大权重。公式：

$$\text{common}_u = \frac{\text{deg}_1(u)}{\text{deg}_1(u) + \text{deg}_2(u)} \cdot \mathbf{e}_1(u) + \frac{\text{deg}_2(u)}{\text{deg}_1(u) + \text{deg}_2(u)} \cdot \mathbf{e}_2(u)$$

这比简单平均更合理，因为交互多的域表示通常更可靠。

### 3.3 监督解耦（Disentangle Layer）

**这是模型的核心创新点。**

把用户表示拆成两部分：
- **domain-common**：跨域通用的兴趣因子（比如喜欢"科幻氛围感"，既体现在科幻电影也体现在电子音乐）
- **domain-specific**：只在当前域有效的兴趣因子（比如只看恐怖电影，但和音乐无关）

**具体实现：投影门控网络**

两个独立的 MLP 分别输出一个 0-1 的门控 mask：

```python
common  = emb * sigmoid(MLP_common(emb))
specific = emb * sigmoid(MLP_specific(emb))
```

关键设计：**软性分离**而非硬切分。门控 mask 让每个维度自主决定归属，保留完整的表达能力。

**约束条件：**
1. **Similarity Loss**：两个域的 common 表示应该相似 → 保证"通用兴趣"真的是通用的
2. **Orthogonality Loss**：common 和 specific 应该正交 → 保证二者信息不冗余

### 3.4 Attention 融合

解耦后得到三个表示：GNN 原始输出、common、specific。怎么合起来做预测？

我设计了一个 **Attention 融合层**：拼接三个表示，过线性层 + softmax 学三个权重，自适应加权。

这比简单 concat 更灵活——有的用户 common 信息更重要（跨域活跃），有的用户 specific 信息更重要（只在单域有独特偏好）。

### 3.5 跨域数据增强

**局部增强**：在同一个域内，把正样本和负样本的表示拆成两半交叉重组，生成新的"混合"样本。迫使模型学会：只有 common 部分匹配是不够的，specific 部分也要匹配才算真正喜欢。

**跨域增强**：把 Domain1 的用户和 Domain2 的负样本配对。增强模型对跨域负样本的区分能力，防止它以为"跨域物品随便推都可以"。

---

## 四、快速开始

### 4.1 环境准备

```bash
cd UnifiedCDR
pip install -r requirements.txt
```

### 4.2 方式A：Mock 数据快速体验（1分钟跑通）

```bash
# 生成模拟数据（字段格式与真实 Amazon 数据完全一致）
python datasets/generate_mock_data.py

# 训练（CPU 2分钟跑完）
python main.py --epoch 10 --gpu -1
```

### 4.3 方式B：真实 Amazon Movie-Music 数据（推荐）

> 真实数据下载可能较慢，如果下载不了可以直接用 Mock 数据面试，逻辑完全一致。

**Step 1：下载原始数据**

从 [Amazon Review Data (2018)](https://jmcauley.ucsd.edu/data/amazon_v2/index.html) 下载：
- `Movies_and_TV.json.gz` → `datasets/raw/Amazon/Movie/`
- `meta_Movies_and_TV.json.gz` → `datasets/raw/Amazon/Movie/`
- `CDs_and_Vinyl.json.gz` → `datasets/raw/Amazon/Music/`
- `meta_CDs_and_Vinyl.json.gz` → `datasets/raw/Amazon/Music/`

**Step 2：格式转换**

原始数据是 JSON 格式，需要先转为 CSV：
```bash
python datasets/convert_json_to_csv.py --domain Movie
python datasets/convert_json_to_csv.py --domain Music
```

**Step 3：K-core 过滤**

```bash
cd datasets
python filter.py --domain Movie
python filter.py --domain Music
```

**Step 4：对齐 ID 并划分训练/验证/测试**

```bash
python process.py --domains Movie-Music
```

**Step 5：训练**

```bash
cd ..
python main.py --epoch 100 --gpu 0
```

### 4.4 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--domains` | 数据集对 | `Movie-Music` |
| `--epoch` | 训练轮数（覆盖 yaml） | `None` |
| `--lr` | 学习率（覆盖 yaml） | `None` |
| `--gpu` | GPU 编号，`-1`=CPU | `0` |
| `--seed` | 随机种子 | `2024` |

---

## 五、文件结构

```
UnifiedCDR/
├── main.py                       # 训练入口（~160行）
├── config.yaml                   # 超参数配置
├── utils.py                      # 数据加载 & 负采样
├── requirements.txt              # 依赖
├── README.md                     # 本文件
├── INTERVIEW.md                  # ⭐ 面试话术（重点看！）
├── models/
│   └── unified_cdr.py            # ⭐ 核心模型（~300行）
└── datasets/
    ├── generate_mock_data.py     # Mock 数据生成器
    ├── convert_json_to_csv.py    # Amazon JSON 转 CSV
    ├── filter.py                 # K-core 过滤
    ├── process.py                # ID 对齐 & 数据划分
    └── utils.py                  # 数据处理工具
```

---

## 六、超参数说明

编辑 `config.yaml` 调整：

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `embedding_size` | GNN 每层输出维度 | 64 |
| `common_dim` | 增强时的"共享-like"维度 | 32 |
| `n_layers` | GNN 层数 | 2 |
| `local_lambda` | 局部增强权重 | 0.5 |
| `cd_lambda` | 跨域增强权重 | 0.5 |
| `cl_sim_weight` | common 一致性损失权重 | 0.01 |
| `cl_org_weight` | 正交损失权重 | 1.0 |

---

## 七、实验结果

### Mock 数据（快速验证）

| 域 | Hit@10 | NDCG@10 |
|----|--------|---------|
| Movie | ~0.22 | ~0.12 |
| Music | ~0.07 | ~0.03 |

> Mock 数据规模较小且较稀疏，指标仅用于验证代码逻辑收敛。真实 Amazon 数据指标会显著提升。

### 真实 Amazon 数据（预期）

参考同领域论文，在 Amazon Movie-Music 上的典型指标范围：

| 域 | Hit@10 | NDCG@10 |
|----|--------|---------|
| Movie | 0.08~0.12 | 0.04~0.07 |
| Music | 0.05~0.09 | 0.03~0.05 |

> 跨域推荐任务的绝对指标通常低于单域，因为数据更稀疏、问题更难。重点看相对提升。

---

## 八、面试建议

1. **先讲业务痛点**：数据稀疏 + 负迁移
2. **再讲技术拆解**：数据层面（增强）+ 表示层面（解耦）
3. **重点讲解耦**：投影门控 + similarity/orthogonality 约束，这是你的核心设计
4. **最后讲落地**：推理延迟同单域，训练可离线

**详细逐字稿和追问答案见 `INTERVIEW.md`**
