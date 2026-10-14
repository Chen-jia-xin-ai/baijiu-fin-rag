# -*- coding: utf-8 -*-
"""③ 章节识别 + 切块 + 表格成块。

本文件承载整个方案里最重要的那条原则:**每个块的嵌入文本自描述**。

    【贵州茅台 600519 | 2025年年度报告 | 第九节 财务报告 / 七、合并财务报表项目注释 | 合并 | 第98页】

这一行同时解决四件事,缺了它后面怎么调都补不回来:
  1. 跨公司对比不串味 —— 公司名进了向量和 BM25 的词表;
  2. 「第几节/第几页」变成可检索字段;
  3. 出处不用猜;
  4. 合并 / 母公司口径可区分(同一个科目两套表,数字不同)。

没有它,一个写着「期末余额 1,234」的块在 12 家公司里向量上无法区分,
所有全景题都会塌向语料占比最大的那家。

章节层级用「字号定候选、正则定层级」:
  - 字号法解决「哪一行是标题」(中报靠换字体——黑体 vs 宋体——比加粗标记可靠);
  - 正则法解决「这是第几级」,**且必须在当前节的作用域内判**——
    「一、」在每一级都复现,不给嵌套作用域根本没法消歧,这就是纯正则不行的原因。
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

log = config.get_logger("03_chunk")

L1_RE = re.compile(r"^第[一二三四五六七八九十]+节")
L2_RE = re.compile(r"^[一二三四五六七八九十]+、")
L3_RE = re.compile(r"^[（(][一二三四五六七八九十]+[）)]")
L4_RE = re.compile(r"^[（(]?\d+[）)、．.]|^\d+、")
SENT_SPLIT_RE = re.compile(r"(?<=[。！？；])|\n")

# 三大报表的标题必须**无条件**当标题,不看字号。
#
# 茅台 FY2025 的「合并资产负债表」和正文同字号同字体(10.4pt SimSun),
# 字号法和换字体法都抓不到它。后果不是少一个标题,而是**上一节的标题一直粘着**:
# 第 58 页的资产负债表块被冠上「第八节 财务报告 / 六、注册会计师对财务报表审计的责任」——
# 那是前面审计报告的二级标题。章节进了嵌入文本、进了 BM25 词表、也进了给用户看的出处,
# 一个错的比没有还坏。报表标题是纯结构标记,正则锚死后不可能误伤(H2 已锚尾)。
STATEMENT_TITLE_RE = re.compile(
    r"^[（(]?\d*[）)、.．]?\s*(合并|母公司)?\s*"
    r"(资产负债表|利润表|现金流量表|所有者权益变动表)"
    r"\s*[（(]?续?[）)]?。?\s*$")

LEVELS = ("l1", "l2", "l3", "l4")


def _body_size(records: list[dict]) -> float:
    """正文字号 = 按字符数加权的众数字号。比取平均稳,标题不会把均值拽偏。"""
    hist: Counter = Counter()
    for r in records:
        for l in r["lines"]:
            hist[l["size"]] += l["nchars"]
    return hist.most_common(1)[0][0] if hist else 10.0


def _body_font(records: list[dict], size: float) -> str:
    hist: Counter = Counter()
    for r in records:
        for l in r["lines"]:
            if abs(l["size"] - size) < 0.1:
                hist[l["font"]] += l["nchars"]
    return hist.most_common(1)[0][0] if hist else ""


def _heading_level(text: str) -> str | None:
    """正则定层级。顺序要紧:先长前缀后短,否则「一、」会抢走「（一）」。"""
    if L1_RE.match(text):
        return "l1"
    if L2_RE.match(text):
        return "l2"
    if L3_RE.match(text):
        return "l3"
    if L4_RE.match(text):
        return "l4"
    return None


def _is_heading(line: dict, body_size: float, body_font: str) -> bool:
    t = line["text"].strip()
    if not t or len(t) > 40:
        return False
    if STATEMENT_TITLE_RE.match(t):
        return True          # 报表标题:字号不可信,正则说了算
    if t[-1] in "。；，,;":
        return False
    if line["size"] >= body_size + 0.5:
        return True
    # 字号相同但换了字体 —— 中文报告里二级标题常这么排
    return bool(body_font) and line["font"] != body_font and line["size"] >= body_size - 0.1 \
        and len(t) <= 30


def _scope_of(*texts: str) -> str:
    """判「合并 / 母公司」口径。同一科目两套表,数字不同,必须区分。

    章节路径常常认不出(# 标题和正文同字号),所以表格块的 caption
    (来自 02 号脚本单独抓的财报标题)是更可靠的判据,一并传进来。
    """
    joined = " ".join(t for t in texts if t)
    if "母公司" in joined:
        return "母公司"
    if "合并" in joined:
        return "合并"
    return ""


def _period_label(meta: dict) -> str:
    return f"{meta['period_year']}年{'年报' if meta['kind'] == 'annual' else '半年报'}"


def _prefix(meta: dict, section: str, page: int, chunk_type: str, unit: str = "") -> str:
    parts = [f"{meta['name']} {meta['code']}", _period_label(meta)]
    if section:
        parts.append(section)
    scope = _scope_of(section)
    if scope:
        parts.append(scope)
    if chunk_type == "table":
        parts.append(f"表格{f'(单位:{unit})' if unit else ''}")
    parts.append(f"第{page}页")
    return "【" + " | ".join(parts) + "】"


def _split_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """按句子边界切,再贪心装到 max_chars。重叠取尾部 overlap 字。"""
    sents = [s for s in SENT_SPLIT_RE.split(text) if s and s.strip()]
    if not sents:
        return []
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) > max_chars:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = tail + s
        else:
            cur += s
        # 单句超长:硬切
        while len(cur) > max_chars * 2:
            chunks.append(cur[:max_chars])
            cur = cur[max_chars - overlap:]
    if cur.strip():
        chunks.append(cur)
    return [c for c in chunks if len(c.strip()) >= 20]


def _split_table(rows_lin: list[str], caption: str, unit: str, cols: list[str],
                 max_chars: int) -> list[str]:
    """超长的表按行切,每一段都重复标题 + 列名,并标「续」。"""
    if not rows_lin:
        return []
    head = " | ".join(p for p in (caption, f"单位:{unit}" if unit else "") if p)
    colhead = f"{head} | 列: " + " | ".join(cols)
    out, cur = [], ""
    for r in rows_lin:
        if cur and len(cur) + len(r) > max_chars:
            out.append(cur)
            cur = colhead + " (续) | " + r
        else:
            cur = (cur + "\n" + r) if cur else (colhead + "\n" + r)
    if cur:
        out.append(cur)
    return out


def chunk_document(meta: dict, records: list[dict],
                   max_chars: int, overlap: int) -> tuple[list[dict], list[dict]]:
    body_size = _body_size(records)
    body_font = _body_font(records, body_size)
    log.info("  %s %s 正文字号=%.1fpt 正文字体=%s", meta["name"], meta["period"],
             body_size, body_font or "(未知)")

    stack = {k: "" for k in LEVELS}
    chunks, table_rows = [], []
    seq = 0

    def section_path() -> str:
        return " / ".join(stack[k] for k in LEVELS if stack[k])

    def emit_text(paragraphs: list[str], page: int):
        nonlocal seq
        text = "\n".join(paragraphs).strip()
        if len(text) < 20:
            return
        section = section_path()
        for piece in _split_text(text, max_chars, overlap):
            seq += 1
            chunks.append({
                **{k: meta[k] for k in ("code", "name", "short", "period", "kind", "period_year", "title")},
                "chunk_id": f"{meta['code']}_{meta['period']}_T{seq:05d}",
                "chunk_type": "text", "page": page, "section": section,
                "scope": _scope_of(section), "unit": "",
                "text": piece,
                "embed_text": f"{_prefix(meta, section, page, 'text')}\n{piece}",
            })

    def emit_table(t: dict, page: int, section: str):
        nonlocal seq
        cols = t["rows"][0] if t["rows"] else []
        for part in _split_table(t["linearized"], t["caption"], t["unit"], cols, config.TABLE_MAX_CHARS):
            seq += 1
            chunks.append({
                **{k: meta[k] for k in ("code", "name", "short", "period", "kind", "period_year", "title")},
                "chunk_id": f"{meta['code']}_{meta['period']}_B{seq:05d}",
                "chunk_type": "table", "page": page, "section": section,
                "scope": _scope_of(section, t["caption"]), "unit": t["unit"],
                "caption": t["caption"], "nrows": t["nrows"], "ncols": t["ncols"],
                "header_from": t["header_from"],
                "text": part,
                "markdown": t["markdown"],
                "embed_text": f"{_prefix(meta, section, page, 'table', t['unit'])}\n{part}",
            })
        # 结构化行 —— 供 pandas 兜底(算术 / 全库聚合,朴素 RAG 做不了的那类题)
        cols = t["rows"][0] if t["rows"] else []
        for r in t["rows"][1:]:
            if not any(r):
                continue
            # 行标签同样是第一个非空格,不是 r[0] —— 见 02 号脚本的 _tidy_columns
            li = next((j for j, c in enumerate(r) if c), 0)
            table_rows.append({
                "code": meta["code"], "name": meta["name"], "period": meta["period"],
                "page": page, "section": section,
                "scope": _scope_of(section, t["caption"]),
                "caption": t["caption"], "unit": t["unit"],
                "label": r[li],
                "values": {cols[j]: r[j] for j in range(li + 1, min(len(r), len(cols))) if r[j]},
            })

    for rec in records:
        page = rec["page"]
        # 把这一页的行和表按纵向位置合成一条事件流,才能保序
        events = [(l["bbox"][1], "line", l) for l in rec["lines"]]
        events += [(t["bbox"][1], "table", t) for t in rec["tables"]]
        events.sort(key=lambda e: e[0])

        # 先把本页标题的位置记下来,好给表格判所属章节(表常出现在标题下方)
        page_headings = [(l["bbox"][1], l["text"].strip()) for l in rec["lines"]
                         if _is_heading(l, body_size, body_font)]
        base_stack = dict(stack)

        def section_at(y: float) -> str:
            st = dict(base_stack)
            for hy, htext in page_headings:
                if hy >= y:
                    break
                lvl = _heading_level(htext) or ("l2" if len(htext) <= 30 else None)
                if lvl:
                    st[lvl] = htext
                    for k in LEVELS[LEVELS.index(lvl) + 1:]:
                        st[k] = ""
            return " / ".join(st[k] for k in LEVELS if st[k])

        buf = []
        for y, kind, obj in events:
            if kind == "table":
                emit_text(buf, page)
                buf = []
                emit_table(obj, page, section_at(y))
                continue
            line = obj
            if _is_heading(line, body_size, body_font):
                emit_text(buf, page)
                buf = []
                lvl = _heading_level(line["text"]) or ("l2" if len(line["text"]) <= 30 else "l3")
                stack[lvl] = line["text"]
                for k in LEVELS[LEVELS.index(lvl) + 1:]:
                    stack[k] = ""
            else:
                buf.append(line["text"])
        emit_text(buf, page)

    return chunks, table_rows


def main() -> int:
    ap = argparse.ArgumentParser(description="章节识别与切块")
    ap.add_argument("--only", default="")
    ap.add_argument("--periods", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    manifest = config.read_jsonl(config.DATA_DIR / "manifest.jsonl")
    if args.only:
        want = {c.strip() for c in args.only.split(",") if c.strip()}
        manifest = [m for m in manifest if m["code"] in want]
    if args.periods:
        want = {p.strip() for p in args.periods.split(",") if p.strip()}
        manifest = [m for m in manifest if m["period"] in want]

    out_chunks = config.CHUNKS_DIR / "chunks.jsonl"
    out_tables = config.TABLES_DIR / "tables.jsonl"
    if out_chunks.exists() and not args.force:
        log.info("已有 %s,用 --force 重跑", out_chunks)

    all_chunks, all_rows = [], []
    for i, meta in enumerate(manifest, 1):
        src = config.PAGES_DIR / f"{meta['code']}_{meta['period']}.jsonl"
        if not src.exists():
            log.warning("[%d/%d] 缺页数据: %s", i, len(manifest), src.name)
            continue
        records = config.read_jsonl(src)
        chunks, rows = chunk_document(meta, records, config.CHUNK_MAX_CHARS, config.CHUNK_OVERLAP_CHARS)
        ntab = sum(1 for c in chunks if c["chunk_type"] == "table")
        l1 = len({c["section"].split(" / ")[0] for c in chunks if c["section"]})
        log.info("[%d/%d] %s %s -> %d 块 (表 %d) | 一级章节 %d 个",
                 i, len(manifest), meta["name"], meta["period"], len(chunks), ntab, l1)
        all_chunks.extend(chunks)
        all_rows.extend(rows)

    with out_chunks.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(config.json_line(c) + "\n")
    with out_tables.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(config.json_line(r) + "\n")

    log.info("=" * 56)
    log.info("共 %d 块(表格块 %d),结构化行 %d 条",
             len(all_chunks), sum(1 for c in all_chunks if c["chunk_type"] == "table"), len(all_rows))
    log.info("输出: %s / %s", out_chunks, out_tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
