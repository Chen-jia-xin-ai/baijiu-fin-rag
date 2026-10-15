# -*- coding: utf-8 -*-
"""⑨ 把 `docs/结论.md` 渲染成一页 PDF。

作业要求附件里的「一页结论」是 **PDF**,仓库里存的是 markdown(便于 diff)。
这一步把 md 转成 md → HTML → PDF,字号和页边距按「**一页装得下**」调过。

环境里没有 pandoc / weasyprint / markdown,所以这里:
  - 用一个只认本仓库用到的子集(标题 / 表格 / 列表 / 粗体 / 行内代码)的极简转换器;
  - 用 playwright 的 chromium `page.pdf()` 出 PDF —— 它已经在跑 06_screenshot.py 时装好了。

    python scripts/09_pdf.py
"""

import argparse
import html
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

CSS = """
@page { size: A4; margin: 9mm 11mm; }
* { box-sizing: border-box; }
body {
  font-family: "Microsoft YaHei", "微软雅黑", -apple-system, sans-serif;
  font-size: 8.7pt; line-height: 1.44; color: #111; margin: 0;
}
h1 { font-size: 14pt; margin: 0 0 5pt; letter-spacing: .01em; }
h2 { font-size: 10pt; margin: 8pt 0 3pt; padding-bottom: 1.5pt;
     border-bottom: 1px solid #d8d8d8; }
p  { margin: 0 0 4pt; }
ul, ol { margin: 0 0 4pt; padding-left: 15pt; }
li { margin: 0 0 2pt; }
hr { border: 0; border-top: 1px solid #e2e2e2; margin: 7pt 0; }
strong { color: #000; }
code { font-family: Consolas, monospace; font-size: 7.9pt;
       background: #f2f2f4; border-radius: 2px; padding: 0 2px; }
table { border-collapse: collapse; width: 100%; margin: 3pt 0 5pt; font-size: 7.9pt; }
th, td { border: 1px solid #ccc; padding: 2.4pt 4pt; vertical-align: top;
         text-align: left; line-height: 1.38; }
th { background: #f4f4f6; font-weight: 600; }
td br { line-height: 1.38; }
em { color: #333; }
blockquote { margin: 3pt 0 4pt 7pt; padding-left: 6pt;
             border-left: 2px solid #d0d0d0; color: #333; }
pre { background: #f7f7f9; border: 1px solid #e2e2e2; border-radius: 3px;
      padding: 3pt 5pt; font-family: Consolas, monospace; font-size: 7.2pt;
      line-height: 1.35; white-space: pre-wrap; }
a { color: #0645ad; text-decoration: none; word-break: break-all; }
"""

# 多页版式(`--flow`):核验报告那种几十页的交付物。一页结论那套字号与页边距
# 是为「压进一页」调出来的,拿它排长文档会挤成一团。
FLOW_CSS = """
@page { size: A4; margin: 12mm 13mm; }
* { box-sizing: border-box; }
body {
  font-family: "Microsoft YaHei", "微软雅黑", -apple-system, sans-serif;
  font-size: 9.6pt; line-height: 1.5; color: #111; margin: 0;
}
h1 { font-size: 16.5pt; margin: 0 0 8pt; letter-spacing: .01em; }
h2 { font-size: 12.5pt; margin: 15pt 0 5pt; padding-bottom: 2pt;
     border-bottom: 1px solid #d8d8d8; page-break-after: avoid; }
h3 { font-size: 10.8pt; margin: 12pt 0 4pt; page-break-after: avoid; }
p  { margin: 0 0 5pt; }
ul, ol { margin: 0 0 5pt; padding-left: 16pt; }
li { margin: 0 0 2.5pt; }
hr { border: 0; border-top: 1px solid #e2e2e2; margin: 9pt 0; }
strong { color: #000; }
code { font-family: Consolas, monospace; font-size: 8.6pt;
       background: #f2f2f4; border-radius: 2px; padding: 0 2px; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0 8pt; font-size: 8.7pt; }
th, td { border: 1px solid #ccc; padding: 3pt 4.5pt; vertical-align: top;
         text-align: left; line-height: 1.42; word-break: break-word; }
th { background: #f4f4f6; font-weight: 600; }
td br { line-height: 1.42; }
tr { page-break-inside: avoid; }
/* 首列几乎都是「Q1 / 序号 / G1 / 名次」这类短标签,折行只会更难看。 */
th:first-child, td:first-child { white-space: nowrap; }
/* 失败码与 chunk_id 是**要照着回查的标识符**,从中间断开等于没法复制。
   让它们不折行,列宽由表格自动布局撑开。 */
td code { white-space: nowrap; }
em { color: #333; }
blockquote { margin: 3pt 0 7pt 8pt; padding-left: 8pt;
             border-left: 3px solid #d0d0d0; color: #333; }
pre { background: #f7f7f9; border: 1px solid #e2e2e2; border-radius: 3px;
      padding: 5pt 7pt; font-family: Consolas, monospace; font-size: 8pt;
      line-height: 1.4; white-space: pre-wrap; page-break-inside: avoid; }
a { color: #0645ad; text-decoration: none; word-break: break-all; }
"""

INLINE = [
    # 链接要排在粗体之前:`[文字](url)` 里可能含 `**`。
    (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), r'<a href="\2">\1</a>'),
    (re.compile(r"\*\*(.+?)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"`(.+?)`"), r"<code>\1</code>"),
]


def inline(text: str) -> str:
    """先转义,再放行我们自己产生的标签 —— 顺序反了会把 <strong> 也转义掉。"""
    out = html.escape(text, quote=False)
    out = out.replace("&lt;br&gt;", "<br>")          # 表格里的换行标记
    for pat, rep in INLINE:
        out = pat.sub(rep, out)
    return out


def cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_sep(line: str) -> bool:
    return bool(re.fullmatch(r"\|[\s:\-\|]+\|", line.strip()))


def convert(md: str, css: str = CSS) -> str:
    lines = md.split("\n")
    out, i = [], 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if not s:
            i += 1
            continue
        if s == "---":
            out.append("<hr>")
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            lv = min(len(m.group(1)), 3)     # 结论 md 只用 h1/h2;核验报告的题号用 h3
            out.append(f"<h{lv}>{inline(m.group(2))}</h{lv}>")
            i += 1
            continue

        # 围栏代码块。必须排在「段落」之前,否则 ``` 那行会被当成普通段落。
        if s.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1                            # 吃掉收尾的 ```
            out.append("<pre>" + html.escape("\n".join(buf)) + "</pre>")
            continue

        # 引用块(模型答案原文)。**内部只做行内语法,不递归块级** ——
        # 模型答案里的「- 项目」若在这里展开成列表,普通连字符也会被当成列表项。
        if s.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                t = lines[i].strip()[1:].strip()
                buf.append(inline(t) if t else "")
                i += 1
            out.append("<blockquote>" + "<br>".join(buf) + "</blockquote>")
            continue

        # 表格:连续的 | 行,第二行是分隔行
        if s.startswith("|") and i + 1 < len(lines) and is_sep(lines[i + 1]):
            head = cells(s)
            i += 2
            body = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                body.append(cells(lines[i]))
                i += 1
            th = "".join(f"<th>{inline(c)}</th>" for c in head)
            tr = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                for r in body)
            out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>")
            continue

        for tag, pat in (("ul", r"^[-*]\s+(.*)$"), ("ol", r"^\d+\.\s+(.*)$")):
            if not re.match(pat, s):
                continue
            items = []
            while i < len(lines):
                mm = re.match(pat, lines[i].strip())
                if mm:
                    items.append(f"<li>{inline(mm.group(1))}</li>")
                    i += 1
                    continue
                # 列表项之间夹空行是合法的「宽松列表」—— 跳过它继续收,
                # 否则每一项会被当成一个独立的 <ol>,序号全变成 1。
                if (not lines[i].strip() and i + 1 < len(lines)
                        and re.match(pat, lines[i + 1].strip())):
                    i += 1
                    continue
                break
            out.append(f"<{tag}>" + "".join(items) + f"</{tag}>")
            break
        else:
            out.append(f"<p>{inline(s)}</p>")
            i += 1
        continue

    return (f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
            f"<style>{css}</style></head><body>{''.join(out)}</body></html>")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(config.DOCS_DIR / "结论.md"))
    ap.add_argument("--out", default=str(config.DOCS_DIR / "结论.pdf"))
    ap.add_argument("--flow", action="store_true",
                    help="多页版式(核验报告这类长文档);默认是一页版式(结论)")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    if not src.exists():
        print(f"找不到 {src}", file=sys.stderr)
        return 1

    doc = convert(src.read_text(encoding="utf-8"), FLOW_CSS if a.flow else CSS)
    # 中间 HTML 写进临时文件,不落在 docs/ 里 —— 交付目录只留 md 和 pdf。
    # 文件名带 out 的 stem,免得两份 PDF 抢同一个临时文件。
    tmp = Path(tempfile.gettempdir()) / f"baijiu_{out.stem}.html"
    tmp.write_text(doc, encoding="utf-8")

    from playwright.sync_api import sync_playwright

    margin = ({"top": "12mm", "bottom": "12mm", "left": "13mm", "right": "13mm"}
              if a.flow else
              {"top": "9mm", "bottom": "9mm", "left": "11mm", "right": "11mm"})
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(tmp.resolve().as_uri())
        pg.emulate_media(media="print")
        pg.pdf(path=str(out), format="A4", print_background=True, margin=margin)
        b.close()

    # 页数:数页对象。用 [^s] 把 `/Type /Pages` 这个父节点排除掉 ——
    # 直接减 count 在只有一页时也能歪打正着,但不如一次判准。
    n = len(re.findall(rb"/Type\s*/Page[^s]", out.read_bytes()))
    print(f"已生成 -> {out}  ({out.stat().st_size/1024:.0f} KB, {n} 页)")
    if n > 1 and not a.flow:
        print(f"⚠ 超过一页({n} 页) —— 作业要求「一页结论」,"
              f"先删内容(结论 md 里保留要点、细节留给 docs/核验报告.md),"
              f"再考虑调小 scripts/09_pdf.py 里 CSS 的 font-size。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
