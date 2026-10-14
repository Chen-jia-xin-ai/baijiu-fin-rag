# -*- coding: utf-8 -*-
"""金标校验 —— 跑评测之前必须先过这一关。

金标写错是这类作业悄悄产出垃圾的最常见方式:题目没问题、检索也没问题,
但「期望值」抄错了一位数,于是正确答案被判成错的,整张表全反。
所以每一道题的金标都要能被**提取出来的原文**证实:

    python eval/verify_gold.py

对 questions.json 里每条 gold 断言 `contain` 里的每个串都出现在
该 (公司, 报告期, 第 N 页) 的提取结果里。比对前把两侧的空格全去掉 ——
PDF 里「2025 年12 月31 日」是常态,空格是排版噪声不是内容。
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

QUESTIONS = Path(__file__).resolve().parent / "questions.json"


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def load_pages(code: str, period: str) -> dict[int, str]:
    """{页码: 该页全部文本(正文 + 表格行,已去空格)}"""
    src = config.PAGES_DIR / f"{code}_{period}.jsonl"
    if not src.exists():
        return {}
    out = {}
    for rec in config.read_jsonl(src):
        parts = [l["text"] for l in rec.get("lines", [])]
        for t in rec.get("tables", []):
            # linearized 是**行列表**(02 号脚本的 _to_rows 返回 list[str]),
            # 03 号脚本直接逐行喂给切块函数。这里别当字符串拼。
            parts.extend(t.get("linearized") or [])
            parts.append(t.get("caption", ""))
            parts.append(t.get("unit", ""))
            for row in t.get("rows", []):
                parts.extend(row)
        out[rec["page"]] = norm(" ".join(parts))
    return out


def main() -> int:
    if not QUESTIONS.exists():
        print(f"缺少 {QUESTIONS}")
        return 1
    qs = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    cache: dict[tuple, dict] = {}
    n_gold = n_ok = 0
    failures = []

    for q in qs:
        golds = q.get("gold") or []
        if not golds:
            print(f"  ⚠ {q['id']:>2} {q['type']:<12} 无 gold,跳过校验")
            continue
        for g in golds:
            n_gold += 1
            key = (g["code"], g["period"])
            if key not in cache:
                cache[key] = load_pages(*key)
            pages = cache[key]
            if not pages:
                failures.append((q["id"], g, "页数据不存在(先跑 02/03)"))
                continue
            # 页码允许写成 [起始, 结束] 表示跨页
            pg = g["page"]
            wanted = pg if isinstance(pg, list) else [pg]
            body = norm(" ".join(pages.get(p, "") for p in wanted))
            miss = [c for c in g.get("contain", []) if norm(c) not in body]
            if miss:
                failures.append((q["id"], g, f"第{wanted}页找不到 {miss}"))
            else:
                n_ok += 1
                print(f"  ✓ {q['id']:>2} {g['code']} {g['period']} 第{wanted}页 "
                      f"{'|'.join(g.get('contain', []))[:60]}")

    print("=" * 60)
    print(f"金标 {n_ok}/{n_gold} 校验通过")
    if failures:
        print("失败明细:")
        for qid, g, why in failures:
            print(f"  ✗ Q{qid} {g['code']} {g['period']} p{g['page']} -> {why}")
        return 1
    print("金标全部可被提取原文证实,可以跑 05_eval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
