# -*- coding: utf-8 -*-
"""⑧ 行级索引:把 tables.jsonl 里九万多条结构化行单独建成一路检索。

**为什么要有它 —— 第一轮评测给出的最刺眼的失效模式不是「跨公司塌陷」,
而是「答案就在那张表里,可表被切成了五块,行标签和它的数字落在了不同块」。**

一张合并资产负债表的线性化文本有两三千字,按 900 字切就是三到四块,
「合同负债」那一行具体掉进哪一块纯属运气。茅台 2025 年末那张表有 40 行、
被切成 5 块(p56/p57/p57续/p58/p59),而提问「合同负债期末余额」时
top-5 里能不能出现**恰好装着那一行**的那一块,测下来是五五开。

修法不是加重排器,是把**粒度降下来**:每一条结构化行本身就是一个可检索单元,
自描述前缀照旧(公司 / 报告期 / 口径 / 页码),嵌入文本就是那一行的线性化。
于是「合并 合同负债 期末余额」直接命中那一行,而不是命中「含有它的某一块」。

顺带一个更要紧的好处:行级单元天然带上 `scope`(合并 / 母公司)和 `period`
两个**可过滤字段**。这正是结论里那句话的落点 —— 「该建库」的意思不是
把事实埋进向量,而是给它结构化字段,让 `合并` 和 `母公司` 从「文风差异」
变成「一个布尔条件」。第一轮评测里 Q4(母公司应付账款)只排到第 6 名,
根因就是口径在向量空间里几乎不可分。

行级块和页级表格块**同形**,所以下游(retriever / 页面 / 评测)一行都不用改。
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import retriever  # noqa: E402

log = config.get_logger("08_rows")

OUT_ROWS = config.INDEX_DIR / "rows.jsonl"
OUT_BM25 = config.INDEX_DIR / "rows_bm25.pkl"
OUT_EMB = config.INDEX_DIR / "rows_emb.npy"

# 二维码、验证码、网址正文page 会被 PyMuPDF 当成正文抓下来。
# 它字数长、字符分布古怪,在向量上反而**很像**「一份密密麻麻的数字表」,
# 于是第一轮评测里今世缘 H1-2026 第 16 页那两串验证码拿到了 BM25 220 分、
# 冲进 Q4 的 top-5。行级索引正好可以顺手把它们拦在门外。
JUNK_RE = __import__("re").compile(r"[A-Za-z0-9+/=._-]{24,}")


def _period_label(period: str) -> str:
    """FY2023 -> 2023年年度报告;H1-2026 -> 2026年半年报。"""
    if period.startswith("H1-"):
        return f"{period[3:]}年半年报"
    return f"{period[2:]}年年度报告"


def build_docs(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """把每条结构化行渲染成一个「只装一行」的表格块。"""
    recs, docs = [], []
    for r in rows:
        vals = r.get("values") or {}
        if not vals:
            # 纯标签行(「(四)所有者权益内部结转」这种只有标题没有数的)留着无益,
            # 它答不了任何查值题,只会稀释召回。
            continue
        body = " | ".join([r["label"]] + [f"{k}={v}" for k, v in vals.items()])
        if JUNK_RE.search(body):
            continue
        head = f"{r['name']} {r['code']} | {_period_label(r['period'])}"
        if r.get("caption"):
            head += f" | {r['caption']}"
        if r.get("scope"):
            head += f" | {r['scope']}"
        if r.get("unit"):
            head += f" | 表格(单位:{r['unit']})"
        head += f" | 第{r['page']}页"
        embed = f"【{head}】\n{body}"
        recs.append({
            "chunk_id": f"{r['code']}_{r['period']}_R{len(recs) + 1:06d}",
            "chunk_type": "row",
            "code": r["code"], "name": r["name"], "period": r["period"],
            "page": r["page"], "section": r.get("section", ""),
            "scope": r.get("scope", ""), "unit": r.get("unit", ""),
            "caption": r.get("caption", ""),
            "text": body, "embed_text": embed,
        })
        docs.append(embed)
    return recs, docs


def main() -> int:
    ap = argparse.ArgumentParser(description="建行级索引")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 行(调试用)")
    ap.add_argument("--no-embed", action="store_true", help="只建词法,跳过向量")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if OUT_ROWS.exists() and not args.force:
        log.info("已有 %s,用 --force 重跑", OUT_ROWS)

    src = config.TABLES_DIR / "tables.jsonl"
    rows = list(config.read_jsonl(src))
    if args.limit:
        rows = rows[: args.limit]
    log.info("读入结构化行 %d 条", len(rows))

    recs, docs = build_docs(rows)
    log.info("留用 %d 条(丢掉 %d 条纯标签行/噪声行)", len(recs), len(rows) - len(recs))

    # ---------- 词法 ----------
    from rank_bm25 import BM25Plus
    t0 = time.time()
    tokens = [retriever.tokenize(d) for d in docs]
    bm25 = BM25Plus(tokens)
    with OUT_BM25.open("wb") as f:
        pickle.dump(bm25, f)
    log.info("BM25 建好,%d 条,%.1fs", len(tokens), time.time() - t0)

    with OUT_ROWS.open("w", encoding="utf-8") as f:
        for c in recs:
            f.write(config.json_line(c) + "\n")

    # ---------- 向量 ----------
    if not args.no_embed:
        from fastembed import TextEmbedding
        log.info("加载 %s …", config.EMBED_MODEL)
        model = TextEmbedding(model_name=config.EMBED_MODEL)
        t0 = time.time()
        emb = np.zeros((len(docs), 512), dtype=np.float32)
        done = 0
        for vec in model.embed(docs, batch_size=config.EMBED_BATCH):
            emb[done] = vec
            done += 1
            if done % 10000 == 0:
                log.info("  已嵌入 %d/%d", done, len(docs))
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb /= norms
        np.save(OUT_EMB, emb)
        log.info("向量建好 %s,%.1fs", emb.shape, time.time() - t0)

    (config.INDEX_DIR / "rows_meta.json").write_text(json.dumps({
        "n_rows": len(recs), "model": config.EMBED_MODEL,
        "has_emb": (not args.no_embed) and OUT_EMB.exists(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("行级索引已落盘 data/index/(rows.jsonl / rows_bm25.pkl / rows_emb.npy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
