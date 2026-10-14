# -*- coding: utf-8 -*-
"""② 从 PDF 提取正文 + 表格,把表格按行列还原。

这一步是整条链路里最容易出错的地方。四个坑都会「安静地给出错误数字」:

坑 1:合并表头断裂。中文年报的财务表大量使用跨列/跨行单元格
      (「本期数」横跨两个子列、「营业收入」纵跨下面一行)。PyMuPDF 的
      extract() 对这类单元格返回**空字符串**,不做传播。结果是「本期/上期」
      与数字脱钩,模型会把数字报在错误的期间下。
      处理:先横向前向填充(跨列),再纵向填充第 0 列(跨行),多级表头拼成
      "本期数 / 营业收入"。

坑 2:**真正的表头常常在 bbox 外面**。合并资产负债表这类整页表格,表头行
      「项目 附注 期末余额 期初余额」被 find_tables 切在表格之外,而表内前几行
      是「衍生金融负债」这种单格行。若把它们当表头,横向前向填充会把
      「衍生金融负债」复制满整行,于是每一列都叫同一个名字,**期末/期初直接错位**。
      处理:单格行绝不算表头;表内没有合格表头时,回到 bbox 上方去找
      一个「空格切分后列数刚好等于表格列数」且含表头关键词的行。

坑 3:单位在表外且各表不一(报表是「元」,管理层讨论常是「万元」)。
      不接上就会「数字对、量级错」。处理:取表上方最近的含「单位」的行,
      没有就继承本报告上一张表的值。

坑 4:页眉混进正文和标题。整页表格的 bbox 紧贴页眉下方,「最近的上一行」
      抓到的就是页眉。处理:按纵向位置把页眉/页码从标题候选和正文里剔掉。
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

log = config.get_logger("02_extract")

UNIT_RE = re.compile(r"单位\s*[:：]\s*([人民币]*\s*[元万亿千百万]*)")
NUM_RE = re.compile(r"^[\s(（\[【]*[-—－–]?[\d,，.]+[)）\]】%]*$")
CJK_RE = re.compile(r"[一-鿿]")
PERIOD_WORD_RE = re.compile(r"(本期|上期|期末|期初|年末|年初|本年|上年|报告期|同比|增减|比上年)")
HEADER_WORDS = ("项目", "附注", "期末", "期初", "本期", "上期", "金额", "余额", "占比",
                "数量", "比例", "序号", "名称", "年末", "年初", "变动", "同比", "单位", "合计")
# 财务报表标题。这三种表名**必须**单独抓:它们是表格最重要的上下文,
# 既决定 caption,又决定「合并 / 母公司」口径(同一科目两套表,数字不同)。
# 而它们的排版极不一致 —— 实测茅台的「合并资产负债表」是 10.4pt/SimSun
# (和正文一模一样,标题检测完全认不出来),「母公司资产负债表」却是 12pt。
# 财报标题。**必须容忍序号前缀**(泸州老窖写「3、合并利润表」、古井贡写「6、母公司现金流量表」),
# 否则标题认不出来,而泸州老窖的标题又偏偏印在**上一页页脚**,于是整张合并利润表
# 拿不到 caption、口径也判成空的。
#
# 但容忍前缀就有误伤的坑:山西汾酒/今世缘有「1、利润表及现金流量表相关科目变动分析表」
# 这种小节标题,前缀一放宽就会被当成财报标题,然后一路粘到后面几十张表上。
# 所以**结尾必须锚死** —— 财报标题到「表」字就结束了,后面还有字的一律不是。
STATEMENT_RE = re.compile(
    r"^[（(]?\d*[）)、.．]?\s*(合并|母公司)?\s*"
    r"(资产负债表|利润表|现金流量表|所有者权益变动表)"
    r"\s*[（(]?续?[）)]?\s*$")
HEAD_BAND = 0.08          # 页眉带:上方 8%。取 6% 会漏掉 y1≈54.7 的页眉(实测)
FOOT_BAND = 0.90          # 页脚带:下方 10%("58 / 143" 在 91% 处,取 6% 会漏)


def _norm_cell(s) -> str:
    if s is None:
        return ""
    s = str(s).replace("\n", " ").replace("　", " ")
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s.replace("（", "(").replace("）", ")")


def _is_numeric(s: str) -> bool:
    s = _norm_cell(s)
    if not s or CJK_RE.search(s):
        return False
    return bool(NUM_RE.match(s)) or bool(re.fullmatch(r"[-—－–]?[\d,，.]+", s))


def _bbox_overlaps(a, b, tol: float = 2.0) -> bool:
    return not (a[2] < b[0] + tol or a[0] > b[2] - tol or
                a[3] < b[1] + tol or a[1] > b[3] - tol)


def page_lines(page) -> list[dict]:
    """把页面拆成行,每行带字号/字体/颜色/bbox —— 供 03 号脚本做标题识别。"""
    out = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            biggest = max(spans, key=lambda s: s.get("size", 0))
            bbox = [round(v, 1) for v in line.get("bbox", (0, 0, 0, 0))]
            # 有些页会把页眉/页码/正文塞进同一条 line(span 文本里带 \n)。
            # 不拆开的话,归一化后每页内容都不同,跨页重复度就抓不到页眉了。
            for part in (p.strip() for p in text.split("\n")):
                if not part:
                    continue
                out.append({
                    "text": part,
                    "size": round(float(biggest.get("size", 0)), 2),
                    "font": biggest.get("font", ""),
                    "color": biggest.get("color", 0),
                    "nchars": len(part),
                    "bbox": bbox,
                })
    return out


def _is_running_head(line: dict, page_h: float) -> bool:
    """页眉:在页面上方窄带里且很短。整页表格的 bbox 紧贴它,不剔掉会污染标题。"""
    return line["bbox"][3] < page_h * HEAD_BAND and len(line["text"]) <= 60


def _is_running_foot(line: dict, page_h: float) -> bool:
    if line["bbox"][1] < page_h * FOOT_BAND:
        return False
    t = line["text"].strip()
    return bool(re.fullmatch(r"[\d\s第页/-]{1,12}", t))


def _is_layout_box(rows: list[list[str]]) -> bool:
    """判断是不是版式框(两栏目录/图注),而不是真表格。"""
    nc = max(len(r) for r in rows)
    cells = [_norm_cell(c) for r in rows for c in r]
    nonempty = [c for c in cells if c]
    if len(nonempty) < 4:
        return True
    if nc == 2:
        per_col = [sum(len(_norm_cell(r[j])) for r in rows if j < len(r)) for j in range(2)]
        total = sum(per_col) or 1
        if max(per_col) / total > 0.82:
            return True
    if sum(len(c) for c in nonempty) / len(nonempty) > 55:   # 平均过长 -> 是段落
        return True
    return False


def _detect_header(rows: list[list[str]]) -> int:
    """返回表内表头行数;0 表示表内没有表头(真表头在上一页或 bbox 外面)。

    两条硬规则:
      - 遇到**单格行**就停:财务表里「衍生金融负债」这种只有一格的续页行标签,
        或「流动资产：」这种小节标签,**都不是数据行**。但注意是「停」不是
        「整表判 0」——否则真表头(项目/附注/期末余额/期初余额)会因为下一行是
        小节标签而被一起丢掉。
      - 遇到「过半是数字」的行就停,那是数据行开始。

    续页的表(第一行就是单格行标签)会得到 0,由跨页继承来补列名。
    """
    n = 0
    for i, r in enumerate(rows[:3]):
        cells = [_norm_cell(c) for c in r if _norm_cell(c)]
        if len(cells) < 2:
            break
        numeric_ratio = sum(1 for c in cells if _is_numeric(c)) / len(cells)
        if numeric_ratio >= 0.5 and not any(PERIOD_WORD_RE.search(c) for c in cells):
            break
        n = i + 1
    return n


def _fill_spanning(grid: list[list[str]], header_rows: int) -> list[list[str]]:
    """修复合并单元格:先横向(跨列),表头多行再纵向拼接,最后纵向填第 0 列(跨行)。"""
    nc = max(len(r) for r in grid)
    grid = [list(r) + [""] * (nc - len(r)) for r in grid]

    if header_rows > 0:
        for r in grid[:header_rows]:
            for j in range(1, nc):
                if not r[j]:
                    r[j] = r[j - 1]
        if header_rows > 1:
            joined = []
            for j in range(nc):
                parts = []
                for r in grid[:header_rows]:
                    if r[j] and r[j] not in parts:
                        parts.append(r[j])
                joined.append(" / ".join(parts))
            grid = [joined] + grid[header_rows:]
            header_rows = 1

    start = header_rows + 1
    for i in range(start, len(grid)):
        if not grid[i][0]:
            grid[i][0] = grid[i - 1][0]
    return grid, header_rows


def _tidy_columns(grid: list[list[str]], header_rows: int):
    """把表头对齐到数据真正所在的列上,并丢掉空列。

    五粮液的合并利润表被 PyMuPDF 判成 **9 列**,而且表头文字和数据本身就错开:

        header : ['', '项目', '', '', '2024 年度', '', '', '2023 年度', '', '']   -> 非空列 1,4,7
        data   : ['', '一、营业总收入', '', '89,175,178,322.70', '', '', '83,272,...', '', '']
                                                                                -> 非空列 1,3,6

    所以「前向填充表头」在这里是**有害**的:它把 '2024 年度' 填到第 3 列上,
    而第 3 列装的是 2024 的数、顶着 '项目' 的名字。结果就是每个数字都挂在错误的列名下面,
    正文检索照样能跑(标签和数字都在同一个块里),**错得很安静**。

    正确做法是反过来:先看数据行里**哪些列真的出现过值**,再给每个这样的列
    挑一个离它最近的非空表头格。列本来整齐时(绝大多数表)这就是恒等映射,不会动任何东西。

    必须在 `_fill_spanning` **之前**调用 —— 之后表头已经被填过,原始位置就丢了。
    """
    if header_rows <= 0 or not grid:
        return grid, header_rows
    n = max(len(r) for r in grid)
    grid = [list(r) + [""] * (n - len(r)) for r in grid]

    # 1) 多行表头并成一行(逐列取各行的非空值拼接),之后按单行表头处理
    if header_rows > 1:
        head = []
        for j in range(n):
            parts = []
            for r in grid[:header_rows]:
                if r[j] and r[j] not in parts:
                    parts.append(r[j])
            head.append(" / ".join(parts))
        grid = [head] + grid[header_rows:]
        header_rows = 1

    used = sorted({j for r in grid[header_rows:] for j in range(n) if r[j]})
    if not used or len(used) == n:
        return grid, header_rows        # 列本来就整齐,原样返回

    hpos = [j for j in range(n) if grid[0][j]]
    names = []
    for k, j in enumerate(used):
        if not hpos:
            names.append(f"列{k + 1}")
            continue
        # 最近的非空格;距离相同时优先取它右边那个(表头常左对齐而落在数据左侧)
        hit = min(hpos, key=lambda h: (abs(h - j), h < j))
        names.append(grid[0][hit] or f"列{k + 1}")

    seen: Counter = Counter()
    for i, nm in enumerate(names):     # 还剩同名就加序号,免得 dict 取值互相覆盖
        seen[nm] += 1
        if seen[nm] > 1:
            names[i] = f"{nm}_{seen[nm]}"

    return [names] + [[r[j] for j in used] for r in grid[header_rows:]], header_rows


def _header_tokens_above(lines: list[dict], table_bbox, ncols: int, page_h: float) -> list[str] | None:
    """表内没有表头时,回到 bbox 上方找一个列数与表格一致的表头行。"""
    cands = [l for l in lines
             if l["bbox"][3] <= table_bbox[1] + 6
             and l["bbox"][1] >= table_bbox[1] - 130
             and not _is_running_head(l, page_h)]
    cands.sort(key=lambda l: l["bbox"][3], reverse=True)
    for l in cands:
        toks = [t for t in re.split(r"[\s|]+", l["text"]) if t]
        if len(toks) == ncols and sum(1 for t in toks if any(w in t for w in HEADER_WORDS)) >= 2:
            return toks
    return None


def _borrow_columns(grid: list[list[str]], want: list[str]):
    """续表继承表头 —— 判据是「承载数据的列数」,不是物理列数。

    一开始这里比的是 `ncols == ncols`,于是白酒这 12 家**全线失守**:
    报表首页被 PyMuPDF 判成 3 列(项目 / 期末余额 / 期初余额),续页却是 4 列、
    9 列 —— 多出来的常是「附注」那一列,首页填了附注号,续页整列空白。

    物理列数一比就「不相等」,于是每一页续表的列名都退化成「列3 / 列4」,
    「合同负债期末是多少」在续页上就再也答不出来:行标签在、数字在,列名没了。
    而资产负债表/利润表动辄跨 2-3 页 —— 也就是最要紧的那几张表的**后半截全废**。

    判据用 `_tidy_columns` 的收缩规则:先看**哪些列真的出现过值**。

    两边的数据列数**仍然可以不等**,而且这恰恰是常态:母公司的「附注」列在首页有值、
    续页整列空白,于是续页的数据列比首页少一个。所以对齐时两端锚定 ——
    第一个数据列对第一个(都是行标签),其余**从右往左**对(期末、期初这种
    价值列永远是靠右的那几列),源里多出来、对不上的那一列(附注)丢掉。
    返回 (列名, 数据行) 或 None。
    """
    if not grid or not want:
        return None
    n = max(len(r) for r in grid)
    grid = [list(r) + [""] * (n - len(r)) for r in grid]
    used = sorted({j for r in grid for j in range(n) if r[j]})
    m, w = len(used), len(want)
    if m < 2 or m > w:
        return None
    # 目标第 k 个数据列的名字:第 0 个取源的第 0 个,其余从右往左数
    names = [want[0]] + [want[w - (m - k)] for k in range(1, m)]
    return names, [[r[j] for j in used] for r in grid]


def _to_markdown(cols: list[str], data_rows: list[list[str]]) -> str:
    nc = len(cols)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * nc]
    for r in data_rows:
        lines.append("| " + " | ".join((list(r) + [""] * nc)[:nc]) + " |")
    return "\n".join(lines)


def _to_rows(cols: list[str], data_rows: list[list[str]], caption: str, unit: str) -> list[str]:
    """行线性化 —— 索引和 LLM 都用这个形式。

    管道表会把「合同负债」和「8,006,739,780.94」隔开几百个 token,
    「合同负债是多少」这种自然语言问题就捞不回来。线性化让行标签紧挨数值。
    """
    head_parts = [p for p in (caption, f"单位:{unit}" if unit else "") if p]
    head = " | ".join(head_parts)
    out = []
    for r in data_rows:
        if not any(r):
            continue
        # 行标签取**第一个非空格**,不写死 r[0]:有些表前面还有一个纯空的列。
        li = next((j for j, c in enumerate(r) if c), None)
        if li is None:
            continue
        pairs = [f"{cols[j]}={r[j]}" for j in range(li + 1, len(r)) if r[j]]
        if not pairs:
            continue
        row = f"项目={r[li]} | " + " | ".join(pairs)
        out.append(f"{head} | {row}" if head else row)
    return out


def _find_context(lines: list[dict], table_bbox, last_unit: str, page_h: float) -> tuple[str, str]:
    """在表上方找「单位:xxx」和表格标题(排除页眉)。"""
    above = [l for l in lines
             if l["bbox"][3] <= table_bbox[1] + 6
             and l["bbox"][1] >= table_bbox[1] - 170
             and not _is_running_head(l, page_h)]
    above.sort(key=lambda l: l["bbox"][3])
    unit, caption = "", ""
    for l in reversed(above):
        if not unit:
            m = UNIT_RE.search(l["text"])
            if m:
                unit = m.group(1).replace(" ", "")
                continue
        if not caption and len(l["text"]) <= 40 and "单位" not in l["text"]:
            caption = l["text"]
    return (unit or last_unit), caption


def extract_page(page, page_no: int, state: dict) -> dict:
    page_h = page.rect.height
    lines = page_lines(page)

    raw_tables = []
    for strategy in ("lines", "text"):
        try:
            found = page.find_tables(strategy=strategy).tables
        except Exception:  # noqa: BLE001
            continue
        raw_tables.extend(found)
        if found:
            break        # lines 已命中就不必再跑 text(慢一倍且易报重复)
    # 同处可能被重复报出,按 bbox 去重
    seen, keep = [], []
    for t in raw_tables:
        tb = list(t.bbox)
        if any(_bbox_overlaps(tb, s, tol=8) for s in seen):
            continue
        seen.append(tb)
        keep.append(t)

    # 本页出现的财报标题(按纵向位置排好),用来给表格定 caption 和口径
    stmt_lines = sorted(
        (l["bbox"][1], l["text"].strip()) for l in lines
        if len(l["text"]) <= 24 and STATEMENT_RE.match(l["text"])
    )

    tables = []
    for t in keep:
        try:
            rows = t.extract()
        except Exception:  # noqa: BLE001
            continue
        if not rows or len(rows) < 2:
            continue
        nc = max(len(r) for r in rows)
        if nc < 2 or _is_layout_box(rows):
            continue

        bbox = [round(v, 1) for v in t.bbox]
        header_rows = _detect_header(rows)
        raw = [[_norm_cell(c) for c in r] for r in rows]
        # 先把表头对齐到数据真正所在的列(必须在 _fill_spanning 之前:之后就丢了原始位置)
        grid, header_rows = _tidy_columns(raw, header_rows)
        grid, header_rows = _fill_spanning(grid, header_rows)
        nc = len(grid[0])

        unit, ctx_caption = _find_context(lines, bbox, state.get("unit", ""), page_h)
        if unit:
            state["unit"] = unit

        above_stmt = [t for y, t in stmt_lines if y < bbox[1]]
        if above_stmt:
            # 本页表格上方就有财报标题 —— 最可靠,直接用它定 caption 和口径
            caption = above_stmt[-1]
            state["statement"] = above_stmt[-1]
            state["statement_page"] = page_no
        elif (bbox[1] < page_h * 0.15 and state.get("statement")
              and page_no - state.get("statement_page", 0) <= 3):
            # 续表页:表格紧贴页顶、本页没有任何标题,只能继承上一页的。
            # **必须卡页距**:口子窖 FY2025 的「合并利润表」在 p77-79,
            # 可它的附注表从 p143 起一路也贴着页顶 —— 没有这个卡子,
            # 后面四十来张明细表的 caption 全变成「合并利润表」,
            # 于是口径被判成「合并」、章节归属也跟着错。
            caption = state["statement"]
        else:
            # 普通表格(管理层讨论里那些):用上面最近的一行小节标题。
            # 这里**不**继承财报标题 —— 否则「母公司现金流量表」会一路粘到
            # 后面几十张附注表上,把口径判反。
            caption = ctx_caption

        if header_rows > 0:
            cols = list(grid[0])
            data_rows = grid[1:]
            header_from = "表内"
        else:
            cols = _header_tokens_above(lines, bbox, nc, page_h)
            data_rows = grid
            header_from = "表外"
            if not cols:
                # 跨页续表:这一页本来就没有表头(表头在上一页),继承上一张表的列名。
                # 合并资产负债表/利润表动辄跨 2-3 页,不做这步续页的列名全是"列N"。
                cont = state.get("carry")
                # 续表还有个 signature:表格紧贴页面顶部开始(上面只有页眉)。
                # 加这条能挡掉「跨页但其实是另一张表」的误继承。
                is_continuation = bbox[1] < page_h * 0.15
                # 列数判据放宽之后(见 _borrow_columns),必须再补一道 ——
                # 「这一页顶上的表」和「上一张有表头的表」得是**同一张报表**,
                # 否则页顶随便一张附注表都会套上资产负债表的列名。
                # 标题相同是最直接的凭据;两边有一个为空则退回位置判据。
                same_stmt = (not caption or not cont.get("caption")
                             or caption == cont["caption"]) if cont else False
                borrowed = None
                if cont and is_continuation and same_stmt and page_no - cont["page"] <= 2:
                    borrowed = _borrow_columns(grid, cont["cols"])
                if borrowed:
                    cols, data_rows = borrowed
                    nc = len(cols)
                    header_from = "跨页继承"
                    if not caption:
                        caption = cont["caption"]
                else:
                    cols = [f"列{j + 1}" for j in range(nc)]
                    header_from = "缺失"

        if header_from != "缺失":
            state["carry"] = {"cols": cols, "ncols": nc,
                              "page": page_no, "caption": caption}

        tables.append({
            "bbox": bbox,
            "caption": caption,
            "unit": unit,
            "nrows": len(grid),
            "ncols": nc,
            "header_from": header_from,
            "rows": [cols] + data_rows,
            "markdown": _to_markdown(cols, data_rows),
            "linearized": _to_rows(cols, data_rows, caption, unit),
        })

    # 本页有财报标题但没表格时也要更新,否则标题会跨不过这一页
    if stmt_lines:
        state["statement"] = stmt_lines[-1][1]
        state["statement_page"] = page_no

    # 坑 4 + 坑 3:把落在表格 bbox 内的行从正文里剔掉,否则同一个数字会在
    # 正文和表格里各出现一次,白占 top-k。
    # 注意这里**不**按位置剔页眉页脚——那个留给 _strip_running 统一做。
    # 早剔会让重复度统计看不到页眉(只剩漂移的那几页),两种机制会互相打架。
    table_boxes = [t["bbox"] for t in tables]
    body = [l for l in lines if not any(_bbox_overlaps(l["bbox"], tb) for tb in table_boxes)]

    text = "\n".join(l["text"] for l in body)
    return {
        "page": page_no,
        "page_h": round(page_h, 1),
        "text": text,
        "lines": body,
        "tables": tables,
        "is_image_only": len(text.strip()) < 20 and bool(page.get_images()),
    }


def _norm_repeat(t: str) -> str:
    """归一化,用于判断「这一行是否每页都出现」:数字统一成 #。"""
    return re.sub(r"\s+", "", re.sub(r"\d+", "#", t))


def _strip_running(records: list[dict], ratio: float = 0.05) -> int:
    """剔除页眉/页脚,返回剔除行数。两条规则并用:

      1. **跨页重复度**:同一行文字(数字归一为 #)出现在 ≥5% 的页面上。
         页眉和页码必然满足,正文几乎不可能——150 页的报告里,同一句话不会
         出现在 8 个不同页面上。
      2. **纵向位置**:落在页面上下窄带里的短行。

    只留规则 2 会漏(页眉 y 坐标有几 pt 漂移,实测 p19/p25/p68 漏了);
    只留规则 1 也会漏(规则 2 先把大部分页眉删了,重复度统计只剩下漂移的那几页,
    反而低于阈值)。所以两种都要,而且统计必须在**未按位置过滤**的行上做。
    """
    npages = len(records)
    if npages < 5:
        return 0

    pages_with: Counter = Counter()
    for r in records:
        for t in {_norm_repeat(l["text"]) for l in r["lines"] if len(l["text"]) <= 60}:
            pages_with[t] += 1

    threshold = max(4, int(npages * ratio))
    running = {t for t, c in pages_with.items() if c >= threshold}

    removed = 0
    for r in records:
        page_h = r.get("page_h") or 842.0
        keep = [l for l in r["lines"]
                if _norm_repeat(l["text"]) not in running
                and not _is_running_head(l, page_h)
                and not _is_running_foot(l, page_h)]
        removed += len(r["lines"]) - len(keep)
        r["lines"] = keep
        r["text"] = "\n".join(l["text"] for l in keep)
        for t in r["tables"]:
            if t["caption"] and _norm_repeat(t["caption"]) in running:
                t["caption"] = ""
            t["linearized"] = _to_rows(t["rows"][0], t["rows"][1:], t["caption"], t["unit"])
    return removed


def extract_pdf(pdf: Path, meta: dict) -> list[dict]:
    doc = pymupdf.open(pdf)
    # state 跨页携带:单位、以及上一张表的列名(供续表继承)
    state: dict = {"unit": "", "carry": None, "statement": ""}
    records = []
    for i in range(doc.page_count):
        try:
            rec = extract_page(doc[i], i + 1, state)
        except Exception as e:  # noqa: BLE001
            log.warning("  p%d 提取异常,跳过: %s", i + 1, e)
            rec = {"page": i + 1, "text": "", "lines": [], "tables": [], "is_image_only": False}
        rec.update({k: meta[k] for k in ("code", "name", "period", "kind", "title")})
        records.append(rec)
    doc.close()
    n_stripped = _strip_running(records)
    if n_stripped:
        log.info("  剔除页眉页脚 %d 行", n_stripped)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="提取 PDF 正文与表格")
    ap.add_argument("--only", default="", help="只跑这几家,逗号分隔代码")
    ap.add_argument("--periods", default="", help="只跑这几期,逗号分隔标签")
    ap.add_argument("--force", action="store_true", help="忽略已有结果重新提取")
    args = ap.parse_args()

    manifest = config.read_jsonl(config.DATA_DIR / "manifest.jsonl")
    if args.only:
        want = {c.strip() for c in args.only.split(",") if c.strip()}
        manifest = [m for m in manifest if m["code"] in want]
    if args.periods:
        want = {p.strip() for p in args.periods.split(",") if p.strip()}
        manifest = [m for m in manifest if m["period"] in want]

    log.info("待提取 %d 份", len(manifest))
    t0 = time.time()
    total_pages = total_tables = 0

    for i, meta in enumerate(manifest, 1):
        pdf = config.BASE_DIR / meta["path"]
        out = config.PAGES_DIR / f"{meta['code']}_{meta['period']}.jsonl"
        if out.exists() and not args.force:
            log.info("[%d/%d] 已有,跳过 %s %s", i, len(manifest), meta["name"], meta["period"])
            continue
        if not pdf.exists():
            log.warning("[%d/%d] PDF 不存在: %s", i, len(manifest), pdf)
            continue

        t = time.time()
        recs = extract_pdf(pdf, meta)
        with out.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(config.json_line(r) + "\n")
        ntab = sum(len(r["tables"]) for r in recs)
        total_pages += len(recs)
        total_tables += ntab
        log.info("[%d/%d] %-6s %-8s %3d页 %4d表 %.1fs%s",
                 i, len(manifest), meta["name"], meta["period"], len(recs), ntab,
                 time.time() - t, "  <-- 表数为0,可疑" if ntab == 0 else "")

    log.info("=" * 56)
    log.info("提取完成: %d 页, %d 张表, 耗时 %.1fs", total_pages, total_tables, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
