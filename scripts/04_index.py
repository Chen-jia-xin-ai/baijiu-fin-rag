# -*- coding: utf-8 -*-
"""④ 建索引:jieba 词表 + BM25Plus + bge-small-zh 向量。

真正的实现在 retriever.py(它同时被评测和网页复用),这里只负责
命令行入口和验收打印 —— 建完索引要能一眼看出「建对了没有」:
块数、维度、公司数、报告期、以及一条真实查询的命中长什么样。
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import retriever  # noqa: E402

log = config.get_logger("04_index")

# 建完索引直接拿这三个问题验一遍。三条分别对应评测里的 Q3 / 跨报告期 / Q5:
# 一条表格查值、一条叙事、一条跨公司(用来确认分路检索真的触发了)。
SMOKE = [
    "贵州茅台2025年合并资产负债表里合同负债的期末余额是多少",
    "贵州茅台2025年年度报告的现金分红预案是什么",
    "山西汾酒和泸州老窖2025年的营业收入哪个更高",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="构建向量 + BM25 索引")
    ap.add_argument("--force", action="store_true", help="忽略缓存,重算分词与向量")
    args = ap.parse_args()

    src = config.CHUNKS_DIR / "chunks.jsonl"
    if not src.exists():
        log.error("缺少 %s,先跑 03_chunk.py", src)
        return 1

    r = retriever.build_index(force=args.force)

    codes = sorted({c["code"] for c in r.chunks})
    periods = sorted({c["period"] for c in r.chunks})
    ntab = sum(1 for c in r.chunks if c["chunk_type"] == "table")
    log.info("=" * 56)
    log.info("块 %d(表格块 %d / 正文块 %d)", len(r.chunks), ntab, len(r.chunks) - ntab)
    log.info("公司 %d 家 | 报告期 %s", len(codes), " ".join(periods))
    log.info("向量 %s | 词表 %s 篇", r.embeddings.shape, len(r.bm25.doc_freqs))

    log.info("-" * 56)
    log.info("检索冒烟(人眼核对页码):")
    for q in SMOKE:
        res = r.search(q, top_k=3)
        tag = f" 分路={res['routed_companies']}" if res["routed_companies"] else ""
        log.info("[%s]%s", q, tag)
        for h in res["hits"]:
            log.info("   #%d %.4f  %s", h["rank"], h["score"], h["citation"])

    (config.INDEX_DIR / "smoke.json").write_text(
        json.dumps([r.search(q, top_k=5) for q in SMOKE], ensure_ascii=False, indent=2),
        encoding="utf-8")
    log.info("索引目录: %s", config.INDEX_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
