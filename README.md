# UnifiedCDR

A unified cross-domain recommendation framework that combines cross-domain data augmentation with supervised disentanglement for alleviating data sparsity and negative transfer.

## Overview

Cross-domain recommendation (CDR) aims to improve recommendation performance in a sparse target domain by leveraging knowledge from a related source domain. However, indiscriminate knowledge transfer can lead to negative transfer, and entangled user representations make it difficult to identify what knowledge is transferable.

UnifiedCDR addresses both issues through:

- **Cross-Domain Augmentation** -- generates additional training signals by recombining disentangled representations across domains
- **Supervised Disentanglement** -- explicitly separates user representations into domain-common and domain-specific components with similarity and orthogonality constraints

## Architecture

```
Domain 1 (Movie)              Domain 2 (Music)
      │                              │
  Embedding Layer               Embedding Layer
      │                              │
  LightGCN Encoder              LightGCN Encoder
      │                              │
  Transfer Layer (degree-weighted user fusion)
      │                              │
  Disentangle Layer (gated projection)
      ├─ common  ──────────────────  ├─ common
      └─ specific                    └─ specific
      │                              │
  Attention Fusion              Attention Fusion
      │                              │
  BPR Loss + Augmentation Loss + Disentanglement Loss
```

## Project Structure

```
UnifiedCDR/
├── main.py                  # Training and evaluation entry point
├── config.yaml              # Hyperparameter configuration
├── utils.py                 # Data loading and negative sampling
├── requirements.txt
├── models/
│   └── unified_cdr.py       # Core model implementation
└── datasets/
    ├── generate_mock_data.py
    ├── convert_json_to_csv.py
    ├── filter.py
    ├── process.py
    └── utils.py
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run with Mock Data

```bash
python datasets/generate_mock_data.py
python main.py --epoch 10 --gpu -1
```

### Run with Amazon Movie-Music

1. Download from [Amazon Review Data (2018)](https://jmcauley.ucsd.edu/data/amazon_v2/index.html):
   - `Movies_and_TV.json.gz` and `meta_Movies_and_TV.json.gz`
   - `CDs_and_Vinyl.json.gz` and `meta_CDs_and_Vinyl.json.gz`

2. Preprocess:

```bash
python datasets/convert_json_to_csv.py --domain Movie
python datasets/convert_json_to_csv.py --domain Music
python datasets/filter.py --domain Movie
python datasets/filter.py --domain Music
python datasets/process.py --domains Movie-Music
```

3. Train:

```bash
python main.py --epoch 100 --gpu 0
```

## Configuration

Key parameters in `config.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `embedding_size` | GNN output dimension per layer | 64 |
| `common_dim` | Disentangled common subspace dimension | 32 |
| `n_layers` | Number of GNN layers | 2 |
| `local_lambda` | Local augmentation loss weight | 0.5 |
| `cd_lambda` | Cross-domain augmentation loss weight | 0.5 |
| `cl_sim_weight` | Common similarity loss weight | 0.01 |
| `cl_org_weight` | Orthogonality loss weight | 1.0 |

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--domains` | Dataset pair | `Movie-Music` |
| `--epoch` | Training epochs (overrides yaml) | None |
| `--lr` | Learning rate (overrides yaml) | None |
| `--gpu` | GPU id, `-1` for CPU | `0` |
| `--seed` | Random seed | `2024` |

## License

This project is for research purposes.
