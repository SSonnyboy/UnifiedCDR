"""
Amazon Review Data JSON → CSV 转换工具
=========================================
Amazon 原始数据是 JSON Lines 格式（.json.gz），
而 filter.py 期望的是 CSV 格式（item,user,rating,timestamp）。
本脚本完成格式转换。

用法:
    python convert_json_to_csv.py --domain Movie
    python convert_json_to_csv.py --domain Music

输入:
    datasets/raw/Amazon/<Domain>/<DomainFullName>.json.gz
    每行格式: {"reviewerID":"...", "asin":"...", "overall":5.0, "unixReviewTime":1370131200}

输出:
    datasets/raw/Amazon/<Domain>/<DomainFullName>.csv
    格式: item,user,rating,timestamp
"""
import os
import gzip
import json
import argparse
from tqdm import tqdm


domain_fullname = {
    'Movie': 'Movies_and_TV',
    'Music': 'CDs_and_Vinyl',
    'Cell': 'Cell_Phones_and_Accessories',
    'Elec': 'Electronics',
}


def convert(domain):
    fullname = domain_fullname[domain]
    input_path = f'./raw/Amazon/{domain}/{fullname}.json.gz'
    output_path = f'./raw/Amazon/{domain}/{fullname}.csv'

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        print(f"Please download {fullname}.json.gz from")
        print(f"  https://jmcauley.ucsd.edu/data/amazon_v2/categoryFiles/{fullname}.json.gz")
        print(f"and place it at {input_path}")
        return

    print(f"Converting {input_path} ...")

    # 先统计行数（用于 tqdm）
    total_lines = 0
    with gzip.open(input_path, 'rt', encoding='utf-8') as fp:
        for _ in fp:
            total_lines += 1

    n_written = 0
    with gzip.open(input_path, 'rt', encoding='utf-8') as fp:
        with open(output_path, 'w') as fout:
            for line in tqdm(fp, total=total_lines, desc='Converting'):
                try:
                    data = json.loads(line.strip())
                    item = data.get('asin', '')
                    user = data.get('reviewerID', '')
                    rating = data.get('overall', 0)
                    timestamp = data.get('unixReviewTime', 0)
                    if item and user:
                        fout.write(f"{item},{user},{rating},{timestamp}\n")
                        n_written += 1
                except (json.JSONDecodeError, KeyError):
                    continue

    print(f"Done! Written {n_written} lines to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, required=True,
                        choices=['Movie', 'Music', 'Cell', 'Elec'])
    args = parser.parse_args()
    convert(args.domain)
