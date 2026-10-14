# -*- coding: utf-8 -*-
"""调试用:在**提取结果**里按关键词找行,用来人工核对表格还原是否正确。

    python scripts/peek.py 合同负债 --code 600519 --period FY2025
    python scripts/peek.py 销售费用 --period FY2025 --tables
    python scripts/peek.py 营业收入 --rows            # 只在结构化行(tables.jsonl)里找

写金标的时候全靠它:金标里的每个数字都必须能在提取结果里指到某公司某期某页,
`eval/verify_gold.py` 会把这条规矩变成硬闸门。
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="在提取结果里查关键词")
    ap.add_argument("keyword")
    ap.add_argument("--code", default="", help="公司代码,逗号分隔")
    ap.add_argument("--period", default="", help="报告期,逗号分隔")
    ap.add_argument("--rows", action="store_true", help="查结构化行 tables.jsonl")
    ap.add_argument("--only-tables", action="store_true", help="只看表格行(不输出正文命中)")
    ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args()

    codes = {c.strip() for c in a.code.split(",") if c.strip()}
    periods = {p.strip() for p in a.period.split(",") if p.strip()}
    pat = re.compile(a.keyword)
    shown = 0

    if a.rows:
        src = config.TABLES_DIR / "tables.jsonl"
        if not src.exists():
            print(f"缺 {src},先跑 03_chunk.py")
            return 1
        for r in config.read_jsonl(src):
            if shown >= a.limit:
                break
            if codes and r["code"] not in codes:
                continue
            if periods and r["period"] not in periods:
                continue
            if not pat.search(r["label"]):
                continue
            print(f"{r['code']} {r['name']} {r['period']} p{r['page']} "
                  f"[{r['scope']}] {r['caption']} 单位={r['unit']}")
            print(f"    {r['label']} -> {r['values']}")
            shown += 1
        print(f"\n共 {shown} 条")
        return 0

    for f in sorted(config.PAGES_DIR.glob("*.jsonl")):
        code, period = f.stem.split("_", 1)
        if codes and code not in codes:
            continue
        if periods and period not in periods:
            continue
        for rec in config.read_jsonl(f):
            if shown >= a.limit:
                break
            for t in rec.get("tables", []):
                if not any(pat.search(c) for r in t["rows"] for c in r):
                    continue
                print(f"[表] {code} {period} p{rec['page']} {t['caption']} "
                      f"单位={t['unit']} 列={t['ncols']} 来源={t['header_from']}")
                print(f"     列名 {t['rows'][0]}")
                for r in t["rows"][1:]:
                    if any(pat.search(c) for c in r):
                        print(f"     {' | '.join(r)}")
                shown += 1
            if a.only_tables:
                continue
            for l in rec.get("lines", []):
                if pat.search(l["text"]):
                    print(f"[正文] {code} {period} p{rec['page']} {l['text'][:120]}")
                    shown += 1
    print(f"\n共 {shown} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
