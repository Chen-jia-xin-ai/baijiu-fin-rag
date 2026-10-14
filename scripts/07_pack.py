# -*- coding: utf-8 -*-
"""⑦ 抽小块样本入库。

语料和索引加起来几百 MB,不可能进 git;但一句「跑 01-04 号脚本就能重建」
对评审不够友好 —— 他得先装 pymupdf、先抓 48 份 PDF 才能看见块长什么样。

所以抽一份**够小的样本**提交:每家公司每期各取文本块和表格块若干,
评审 clone 下来直接就能看出「自描述前缀 + 行线性化」到底长什么样,
也能拿它跑 `retriever.py` 之外的所有纯函数。

    python scripts/07_pack.py --per 3
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def sample(rows: list[dict], per: int, keyf, seed: int = 20260101, rng: random.Random = None) -> list[dict]:
    """按 key 分组,每组随机取 per 条。随机是为了样本里不全是每份报告的第一页。"""
    rng = rng or random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[keyf(r)].append(r)
    out = []
    for k in sorted(groups):
        g = groups[k]
        out.extend(g if len(g) <= per else rng.sample(g, per))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="抽样本块")
    ap.add_argument("--per", type=int, default=3, help="每家每期各类型取几块")
    a = ap.parse_args()

    chunks = config.read_jsonl(config.CHUNKS_DIR / "chunks.jsonl")
    rows = config.read_jsonl(config.TABLES_DIR / "tables.jsonl")
    if not chunks:
        print("先跑 03_chunk.py")
        return 1

    rng = random.Random(20260101)
    c_sample = sample(chunks, a.per,
                      lambda c: (c["code"], c["period"], c["chunk_type"]), rng=rng)
    r_sample = sample(rows, max(a.per, 2), lambda r: (r["code"], r["period"]), rng=rng)

    out_c = config.CHUNKS_DIR / "sample.jsonl"
    out_t = config.TABLES_DIR / "sample.jsonl"
    config.write_jsonl(out_c, c_sample)
    config.write_jsonl(out_t, r_sample)

    kb = lambda p: p.stat().st_size / 1024
    print(f"块样本 {len(c_sample)}/{len(chunks)} 条 -> {out_c} ({kb(out_c):.0f} KB)")
    print(f"行样本 {len(r_sample)}/{len(rows)} 条 -> {out_t} ({kb(out_t):.0f} KB)")
    print("两份样本已在 .gitignore 里开了例外,可以直接提交。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
