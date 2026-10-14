# -*- coding: utf-8 -*-
"""⑤ 评测:10 题 × 4 种检索模式,逐题记录召回、判分、失败码。

判分分三层,按可信度从高到低:

  1. **检索命中(全自动,不碰 LLM)。** 金标是 `(公司, 报告期, 页码)`,
     不是块 id —— 重新切块后金标依然成立。记 recall@5 / MRR / 金标名次。
  2. **确定性判分。** 答案是数字或实体的题,正则取出数字按容差比、
     实体按别名精确匹配。无评审、无偏置。
  3. **LLM 评审。** 只用于自由文本题。**必须写明:deepseek-chat 在评
     deepseek-chat,存在自我偏好**,所以另留一列人工复核。

逐题记录的重点是**区分「检索失败」和「阅读失败」**:
  - 金标不在 top-10          -> 切块 / 嵌入的问题
  - 金标在 top-10 但没进 top-5 -> 重排器能救
  - 金标在前 3 但答案错       -> 生成的问题(或口径 / 单位串了)
这样一页结论里的「题型 × 失败码」矩阵才是长出来的,不是编出来的。
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import llm  # noqa: E402
import retriever  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
QUESTIONS = EVAL_DIR / "questions.json"
RUNS = EVAL_DIR / "runs"

# 七种检索配置。前四种是第一轮的基线(可复现),后三种是这一轮的改动。
# 「hybrid-row」并进行级索引;「hybrid-row+filter」再把公司和报告期当**硬条件**。
MODES = [
    {"name": "hybrid",            "mode": "hybrid",     "route": True},
    {"name": "hybrid-noroute",    "mode": "hybrid",     "route": False},
    {"name": "vector",            "mode": "vector",     "route": True},
    {"name": "bm25",              "mode": "bm25",       "route": True},
    {"name": "row",               "mode": "row",        "route": True},
    {"name": "hybrid-row",        "mode": "hybrid-row", "route": True},
    {"name": "hybrid-row+filter", "mode": "hybrid-row", "route": True, "hard_filter": True},
]
BASE = "hybrid"                 # 基线:第一轮就是它,前后可比
BEST = "hybrid-row+filter"      # 这一轮的改进版
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def gold_rank(hits: list[dict], gold: dict) -> int | None:
    """金标在命中列表里的名次(1-based),没命中返回 None。

    页码允许写成列表(跨页表),也允许 ±1 容差(表和叙事段常跨页断开)。
    """
    pages = gold["page"] if isinstance(gold["page"], list) else [gold["page"]]
    tol = gold.get("page_tol", 1)
    for h in hits:
        if h["code"] != gold["code"] or h["period"] != gold["period"]:
            continue
        if any(abs(h["page"] - p) <= tol for p in pages):
            return h["rank"]
    return None


def recall_at(hits: list[dict], golds: list[dict], k: int) -> tuple[int, int]:
    hit = sum(1 for g in golds if (r := gold_rank(hits, g)) and r <= k)
    return hit, len(golds)


def extract_numbers(text: str) -> list[float]:
    out = []
    for m in NUM_RE.finditer(text or ""):
        s = m.group(0).replace(",", "").strip(".")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def _has_number(answer: str, want: float, tol: float) -> bool:
    for v in extract_numbers(answer):
        for cand in (v, v * 1e8, v / 1e8, v * 1e4, v / 1e4):  # 允许亿元/万元换算
            if abs(cand - want) <= max(abs(want) * tol, 1e-6):
                return True
    return False


def numeric_verdict(answer: str, gold: dict) -> str | None:
    """确定性判分。金标给了 numeric 就比数字,给了 entity 就比实体名。

    `numeric` 允许写成**列表**,此时要求每个数都出现 —— 跨公司对比题必须这样,
    否则「只答了汾酒、压根没提泸州老窖」也会被判成对。
    """
    if not answer:
        return None
    if "numeric" in gold:
        wants = gold["numeric"]
        wants = wants if isinstance(wants, list) else [wants]
        tol = float(gold.get("tolerance", 0.01))
        if all(_has_number(answer, float(w), tol) for w in wants):
            return "CORRECT"
        if any(_has_number(answer, float(w), tol) for w in wants):
            return "PARTIAL"
        return "WRONG"
    if "entity" in gold:
        ok = any(e in (answer or "") for e in gold["entity"])
        return "CORRECT" if ok else "WRONG"
    return None


# 「检索类」失败码:东西压根没捞出来,或者捞全了也没用(聚合/排序题)。
# 「阅读类」失败码:东西就摆在 top-5 里,模型读错了、读漏了、或读对了不算。
# 这个区分是整个评测最有用的那一刀 —— 它决定下一步该修索引还是该修 prompt。
READING_CODES = {"UNIT_ERROR", "SCOPE_CONFUSION", "PERIOD_CONFUSION",
                 "TABLE_STRUCTURE_LOSS", "READ_ERROR", "HALLUCINATION"}


def infer_failure(q: dict, hits: list[dict], golds: list[dict], verdict: str) -> str:
    """按证据推断失败码。金标位置决定「检索 or 阅读」,不是靠猜。"""
    if verdict == "CORRECT":
        return ""
    if verdict == "NO_ANSWER":
        return "NO_ANSWER"
    if not golds:
        # 金标故意留空的题(全库聚合),它的失败本来就是聚合失败
        return "AGGREGATION_FAIL"
    found = [r for r in (gold_rank(hits, g) for g in golds) if r]
    if not found:
        return "RETRIEVAL_MISS"
    if verdict == "REFUSED":
        # 金标就在眼前,模型还说「现有材料不足以回答」—— 这不是检索的锅。
        # 第一轮评测里 8/10 题落到这里,是整份报告里最值得写的一条。
        return "READ_ERROR"
    if min(found) > 5:
        return "RETRIEVAL_MISS"
    code = q.get("expect_fail", "")
    return code if code in READING_CODES else "READ_ERROR"


# 附注号那一列不是金额,取「本期数」时必须跳过
NOTE_COL_RE = re.compile(r"附注|注释|^注$|备注")


def _first_amount(values: dict) -> float | None:
    """取一行里**第一个金额**。

    不能写死列名。同一个「本期数」在 12 家年报里叫法全不一样:
    茅台写 `2025 年度`,古井贡和洋河写 `列4`(表头没识别出来只剩个位置),
    管理层讨论里又写成 `本期数`。但它们有个共同点 ——
    **在行字典里都排在「上期数」前面**(列顺序就是 dict 的插入顺序),
    而附注号是唯一需要排除的非金额列。
    """
    for k, v in values.items():
        if NOTE_COL_RE.search(k):
            continue
        try:
            return float(str(v).replace(",", "").strip())
        except ValueError:
            continue
    return None


def pandas_fallback(qid: int) -> dict | None:
    """第三桶兜底:结构化行已经在 tables.jsonl 里,聚合题用 pandas 秒算。

    这不是「RAG 的补丁」,而是结论的一部分:同一批材料,
    带日期的事实归检索库,算术和全库聚合归表格工具。
    """
    src = config.TABLES_DIR / "tables.jsonl"
    if not src.exists():
        return None
    rows = config.read_jsonl(src)

    def pick(period, codes, label_any, cap_any):
        """挑出一行:{code: (name, page, label, value)}。同一 code 出现多次取第一个。"""
        out = {}
        for r in rows:
            if r["period"] != period or r["scope"] != "合并":
                continue
            if codes and r["code"] not in codes:
                continue
            cap = r["caption"] or ""
            if not any(k in cap for k in cap_any):
                continue
            if not any(k in r["label"] for k in label_any):
                continue
            if r["code"] in out:
                continue
            v = _first_amount(r["values"])
            if v is not None:
                out[r["code"]] = (r["name"], r["page"], r["label"], v)
        return out

    if qid == 7:
        # 12 家里 2025 年销售费用率最低的是谁。
        # 分母先试「其中:营业收入」(最纯的营业收入),没有就退回营业总收入 ——
        # 茅台这两个数差 32 亿(营业总收入含利息收入),换口径会挪动小数点后一位。
        exp = pick("FY2025", None, ["销售费用"], ["利润表"])
        rev = pick("FY2025", None, ["其中：营业收入"], ["利润表"]) \
            or pick("FY2025", None, ["营业收入", "营业总收入"], ["利润表"])
        if not exp or not rev:
            return None
        out = []
        for code in sorted(exp):
            if code not in rev:
                continue
            e, r = exp[code], rev[code]
            out.append({"code": code, "name": e[0], "page": e[1], "分母科目": r[2],
                        "销售费用": e[3], "营业收入": r[3],
                        "销售费用率%": round(e[3] / r[3] * 100, 2)})
        out.sort(key=lambda x: x["销售费用率%"])
        return {"method": "pandas over data/tables/tables.jsonl(单位:元)",
                "分母": "优先「其中:营业收入」,缺失时用「营业总收入」",
                "结果": out}

    if qid in (6, 8):
        out = []
        for r in rows:
            if r["scope"] != "合并" or "合同负债" not in r["label"]:
                continue
            if "资产负债" not in (r["caption"] or ""):
                continue
            amt = _first_amount(r["values"])
            if amt is None:
                continue
            out.append({"code": r["code"], "name": r["name"], "period": r["period"],
                        "page": r["page"], "本期数": amt})
        out.sort(key=lambda x: (x["code"], x["period"]))
        return {"method": "pandas over data/tables/tables.jsonl", "结果": out}
    return None


def _hit_rows(hits: list[dict]) -> list[dict]:
    return [{"rank": h["rank"], "score": h["score"], "score_vector": h["score_vector"],
             "score_bm25": h["score_bm25"], "chunk_id": h["chunk_id"],
             "chunk_type": h["chunk_type"], "citation": h["citation"],
             "snippet": h["text"][:80]} for h in hits]


def _judge(r: retriever.Retriever, q: dict, spec: dict, golds: list[dict]) -> dict:
    """跑一个模式并判分。返回 {hits, answer, verdict, judge, gold_rank}。

    判分优先级:确定性(数字/实体)> LLM 评审。
    确定性判分只读 `golds[0]` —— 跨公司题的完整金标因此挂在第一条上(见 questions.json)。
    """
    res = r.search(q["question"], top_k=5, mode=spec["mode"], route=spec["route"],
                   hard_filter=spec.get("hard_filter", False))
    hits = res["hits"]
    out = {"hits": _hit_rows(hits), "routed": res["routed_companies"],
           "filters": res.get("filters", {}), "answer": None, "verdict": "",
           "judge": {}, "gold_rank": None, "_raw": hits}
    found = [x for x in (gold_rank(hits, g) for g in golds) if x]
    out["gold_rank"] = min(found) if found else None

    gen = llm.answer(q["question"], hits)
    ans = gen.get("answer") or ""
    out["answer"] = ans
    out["answer_note"] = gen.get("note", "")

    det = numeric_verdict(ans, golds[0]) if golds else None
    out["deterministic"] = det
    if det:
        out["verdict"] = det
        out["judge"] = {"verdict": det, "reason": "确定性判分(数字容差 / 实体匹配)"}
    elif ans and golds:
        j = llm.judge(q["question"], q.get("gold_answer", ""), ans)
        out["judge"] = j
        out["verdict"] = j.get("verdict", "")
    else:
        out["verdict"] = "REFUSED" if ans else "NO_ANSWER"
        out["judge"] = {"verdict": out["verdict"], "reason": gen.get("note", "")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="跑评测")
    ap.add_argument("--no-llm", action="store_true", help="只算检索指标,不调模型")
    ap.add_argument("--modes", default="", help="逗号分隔,默认七种全跑")
    args = ap.parse_args()

    qs = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    r = retriever.load()
    modes = MODES
    if args.modes:
        want = {m.strip() for m in args.modes.split(",") if m.strip()}
        modes = [m for m in MODES if m["name"] in want]
    base_spec = next(m for m in MODES if m["name"] == BASE)
    best_spec = next(m for m in MODES if m["name"] == BEST)
    if not r.rows:
        print("!! 没找到行级索引(data/index/rows_bm25.pkl),"
              "row / hybrid-row 会退化。先跑 scripts/08_rows.py", flush=True)

    records = []
    for q in qs:
        golds = q.get("gold") or []
        rec = {"id": q["id"], "type": q["type"], "question": q["question"],
               "cross_company": q.get("cross_company", False),
               "expect_fail": q.get("expect_fail", ""),
               "modes": {}, "hits": [], "answer": None, "verdict": "", "failure": "",
               "n_gold": len(golds), "gold_rank": None}

        for spec in modes:
            res = r.search(q["question"], top_k=10, mode=spec["mode"], route=spec["route"],
                           hard_filter=spec.get("hard_filter", False))
            hits = res["hits"]
            h5, n = recall_at(hits, golds, 5) if golds else (0, 0)
            ranks = [gold_rank(hits, g) for g in golds] if golds else []
            found = [x for x in ranks if x]
            rec["modes"][spec["name"]] = {
                "recall@5": [h5, n], "mrr": round((1.0 / min(found)) if found else 0.0, 3),
                "gold_ranks": ranks, "routed": res["routed_companies"],
                "filters": res.get("filters", {}), "top10": _hit_rows(hits),
            }
            if spec["name"] == BASE:
                rec["hits"] = rec["modes"][BASE]["top10"][:5]

        if not args.no_llm:
            base = _judge(r, q, base_spec, golds)
            best = _judge(r, q, best_spec, golds) if best_spec["name"] != BASE else base
            rec["base"] = base
            rec["best"] = best
            # 报告主体沿用基线口径,另开一列放改进后
            rec["answer"] = base["answer"]
            rec["answer_note"] = base.get("answer_note", "")
            rec["verdict"] = base["verdict"]
            rec["judge"] = base["judge"]
            rec["gold_rank"] = base["gold_rank"]
            # 失败码按**改进后**的召回判 —— 报告要回答的是「修完还差在哪」
            rec["failure"] = infer_failure(q, best["_raw"], golds, best["verdict"])
            for k in ("base", "best"):
                rec[k].pop("_raw", None)

        if q.get("pandas_fallback"):
            rec["pandas"] = pandas_fallback(q["id"])

        f1 = {"CORRECT": "✅", "PARTIAL": "⚠️", "WRONG": "❌", "REFUSED": "🚫"}.get(rec["verdict"], "·")
        if args.no_llm:
            print(f"  Q{q['id']:>2} {q['type']:<12} 金标名次={rec['gold_rank']}", flush=True)
        else:
            f2 = {"CORRECT": "✅", "PARTIAL": "⚠️", "WRONG": "❌", "REFUSED": "🚫"}.get(
                rec["best"]["verdict"], "·")
            print(f"  Q{q['id']:>2} {q['type']:<12} 前={f1}{rec['verdict']:<8} "
                  f"后={f2}{rec['best']['verdict']:<8} {rec['failure']}", flush=True)
        records.append(rec)

    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    (RUNS / f"{stamp}.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(records, modes, r, stamp, args.no_llm)
    print(f"\n写出 eval/results.md 与 eval/runs/{stamp}.json")
    return 0


def write_report(records, modes, r: retriever.Retriever, stamp: str, no_llm: bool) -> None:
    L = []
    A = L.append
    n_tab = sum(1 for c in r.chunks if c["chunk_type"] == "table")
    A("# 白酒财报问答知识库 · 评测结果\n")
    A(f"语料:白酒 12 家 × 4 期(2023/2024/2025 年报 + 2026 半年报)。")
    A(f"索引:{len(r.chunks):,} 个页级块(其中表格块 {n_tab:,})+ "
      f"{len(r.rows):,} 个行级块;检索前剔除噪声块 {r.n_junk} 个(二维码 / 目录点线)。")
    A(f"生成模型:`{config.DEEPSEEK_MODEL}`;嵌入模型:`{config.EMBED_MODEL}`。"
      f"运行时间戳 `{stamp}`。\n")
    A("判分口径:检索命中看 `(公司, 报告期, 页码)`,与切块方式解耦;"
      "答案判分先走确定性规则(数字容差 / 实体匹配),命中不了才交给 LLM 评审。\n")

    # —— 检索指标表 ——
    A("## 一、检索模式消融(recall@5 / MRR)\n")
    A("前四种是第一轮的基线,后面三种是本轮的改动,列在同一张表里才好比。\n")
    A("| 模式 | 说明 | recall@5 命中/应有 | 平均 MRR | 金标 10 名内命中 |")
    A("|---|---|---|---|---|")
    DESC = {
        "hybrid": "基线:页级块 向量+BM25",
        "hybrid-noroute": "基线:同上,关掉分路检索",
        "vector": "基线:仅向量",
        "bm25": "基线:仅 BM25",
        "row": "仅行级块(66k 条结构化行)",
        "hybrid-row": "页级 + 行级,四路 RRF",
        "hybrid-row+filter": "★ 再把公司/报告期当硬过滤条件",
    }
    for spec in modes:
        name = spec["name"]
        tot_h = tot_n = 0
        mrrs, in10 = [], 0
        for rec in records:
            m = rec["modes"][name]
            tot_h += m["recall@5"][0]
            tot_n += m["recall@5"][1]
            mrrs.append(m["mrr"])
            if any(x for x in m["gold_ranks"]):
                in10 += 1
        A(f"| {name} | {DESC.get(name,'')} | {tot_h}/{tot_n} | "
          f"{sum(mrrs)/max(len(mrrs),1):.3f} | {in10}/{len(records)} |")
    A("")

    # —— 逐题表 ——
    A("## 二、逐题记录\n")
    if no_llm:
        A("| # | 类型 | 问题 | 金标名次(基线) | 金标名次(hybrid-row+filter) |")
        A("|---|---|---|---|---|")
        for rec in records:
            q = rec["question"].replace("|", "丨")
            m = rec["modes"][BEST]
            f = [x for x in m["gold_ranks"] if x]
            A(f"| {rec['id']} | {rec['type']} | {q} | {rec['gold_rank'] or '未命中'} | "
              f"{min(f) if f else '未命中'} |")
    else:
        A("「前 / 后」= 基线 `hybrid` 与改进版 `hybrid-row+filter` 的答案判分。"
          "失败码按**改进后**的召回推断。\n")
        A("| # | 类型 | 问题 | 金标名次(前→后) | 判分(前) | 判分(后) | 失败码 |")
        A("|---|---|---|---|---|---|---|")
        icon = {"CORRECT": "✅", "PARTIAL": "⚠️", "WRONG": "❌", "REFUSED": "🚫", "NO_ANSWER": "∅"}
        for rec in records:
            q = rec["question"].replace("|", "丨")
            mb = rec["modes"][BEST]
            fb = [x for x in mb["gold_ranks"] if x]
            rk = f"{rec['gold_rank'] or '—'}→{min(fb) if fb else '—'}"
            A(f"| {rec['id']} | {rec['type']} | {q} | {rk} | "
              f"{icon.get(rec['verdict'],'·')} {rec['verdict']} | "
              f"{icon.get(rec['best']['verdict'],'·')} {rec['best']['verdict']} | "
              f"`{rec['failure'] or '—'}` |")
    A("")

    # —— 题型 × 失败码矩阵 ——
    A("## 三、题型 × 失败码\n")
    types = sorted({rec["type"] for rec in records})
    codes = sorted({rec["failure"] for rec in records if rec["failure"]})
    if codes:
        A("| 题型 \\ 失败码 | " + " | ".join(codes) + " |")
        A("|---" * (len(codes) + 1) + "|")
        for t in types:
            row = []
            for c in codes:
                n = sum(1 for rec in records if rec["type"] == t and rec["failure"] == c)
                row.append(str(n) if n else "·")
            A(f"| {t} | " + " | ".join(row) + " |")
    else:
        A("(本轮无失败码)\n")
    A("")

    # —— 逐题明细 ——
    A("## 四、逐题明细(召回块 / 模型答案 / 判分理由)\n")
    for rec in records:
        A(f"### Q{rec['id']} · {rec['type']}")
        A(f"**问题**:{rec['question']}\n")
        if rec.get("cross_company"):
            A(f"> 跨公司题。分路检索:基线 {rec['modes']['hybrid']['routed'] or '未触发'}"
              f" → 改进后 {rec['modes'][BEST]['routed'] or '未触发'}\n")
        A("**基线 `hybrid` 召回的 top-5**:\n")
        A("| 名次 | 融合分 | 向量 | BM25 | 出处 | 片段 |")
        A("|---|---|---|---|---|---|")
        for h in rec["hits"]:
            A(f"| {h['rank']} | {h['score']} | {h['score_vector']} | {h['score_bm25']} | "
              f"{h['citation']} | {h['snippet'].replace('|', '丨')}… |")
        A("")
        # 改进后的召回只列出处 —— 它是「修完能不能够到金标」的证据,片段已在上面出现过
        mb = rec["modes"][BEST]
        fl = mb["filters"]
        A(f"**`{BEST}` 召回的 top-5**"
          f"(硬过滤:公司={fl.get('codes') or '不限'} 报告期={fl.get('periods') or '不限'}):\n")
        A("| 名次 | 融合分 | 出处 | 类型 |")
        A("|---|---|---|---|")
        for h in mb["top10"][:5]:
            A(f"| {h['rank']} | {h['score']} | {h['citation']} | {h['chunk_type']} |")
        A("")
        if rec.get("answer"):
            A(f"**模型答案(基线)**:\n\n> {rec['answer'].replace(chr(10), chr(10) + '> ')}\n")
        elif rec.get("answer_note"):
            A(f"**未生成**:{rec['answer_note']}\n")
        j = rec.get("judge") or {}
        if j:
            A(f"**判分(基线)**:`{j.get('verdict','')}` — {j.get('reason','')}")
            if "确定性" not in j.get("reason", ""):
                A("")
                A("_评审由 `deepseek-chat` 完成,而被评答案同样出自 `deepseek-chat`,"
                  "**存在自我偏好**,故本列仅供参考。_\n")
        if not no_llm:
            b = rec.get("best") or {}
            A(f"**判分({BEST})**:`{b.get('verdict','')}` "
              f"— {(b.get('judge') or {}).get('reason','')}\n")
        if rec.get("pandas"):
            A("**第三桶兜底(pandas)**:")
            A("```json")
            A(json.dumps(rec["pandas"], ensure_ascii=False, indent=2)[:1500])
            A("```\n")
        A("")

    # —— 改进前后汇总 ——
    if not no_llm:
        A("## 五、改进前后\n")
        cnt = {"CORRECT": 0, "PARTIAL": 0, "WRONG": 0, "REFUSED": 0, "NO_ANSWER": 0}
        cnt2 = dict(cnt)
        for rec in records:
            cnt[rec["verdict"]] = cnt.get(rec["verdict"], 0) + 1
            v2 = rec["best"]["verdict"]
            cnt2[v2] = cnt2.get(v2, 0) + 1
        A(f"- 基线 `hybrid`:答对 **{cnt['CORRECT']}/10**,部分对 {cnt['PARTIAL']},"
          f"错 {cnt['WRONG']},拒答 {cnt['REFUSED']}。")
        A(f"- 改进后 `{BEST}`:答对 **{cnt2['CORRECT']}/10**,部分对 {cnt2['PARTIAL']},"
          f"错 {cnt2['WRONG']},拒答 {cnt2['REFUSED']}。")
        # 只看 CORRECT 会漏掉「部分对 → 错」这种**退化**。按好坏给判分排序,
        # 任何方向的变化都要露出来 —— 尤其要暴露改进引入的**新**错误。
        RANK = {"CORRECT": 3, "PARTIAL": 2, "WRONG": 1, "REFUSED": 0, "NO_ANSWER": 0}
        better, worse = [], []
        for rec in records:
            a, b = RANK.get(rec["verdict"], 0), RANK.get(rec["best"]["verdict"], 0)
            if b > a:
                better.append(f"{rec['id']}({rec['verdict']}→{rec['best']['verdict']})")
            elif b < a:
                worse.append(f"{rec['id']}({rec['verdict']}→{rec['best']['verdict']})")
        A(f"- 变好的题:{'、'.join(better) if better else '无'};"
          f"**变差的题:{'、'.join(worse) if worse else '无'}**。")
        A("")
        A("两次运行的完整逐题快照分别存在 `eval/results_v1.md` 与本次的 `eval/runs/*.json` 里,"
          "上面的对比可以逐条回查。\n")

    A("## 六、判分方法的偏置说明\n")
    A("- **检索命中**:全自动、无偏置。金标是页码而非块 id,重新切块后依然可比。")
    A("- **确定性判分**:数字题按容差比(允许亿元/万元换算)、实体题按别名比,无评审、无偏置。")
    A(f"- **LLM 评审**:`{config.DEEPSEEK_MODEL}` 评 `{config.DEEPSEEK_MODEL}`,**自我偏好不可避免**;"
      f"温度 0,金标进 prompt,四档输出。人工复核列留空由本人填写。")
    A("- 题目与金标**全部手工撰写**,未用 LLM 生成 —— 否则评测是循环论证。"
      "`eval/verify_gold.py` 会断言每个金标数字确实出现在它标注的那一页里;"
      "但它只能验**存在性**,验不了**语义** —— Q2 的金标就是这么错过去的,详见该题 `_note`。")

    (EVAL_DIR / "results.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
