# -*- coding: utf-8 -*-
"""① 从巨潮下载白酒 12 家的年报 / 半年报。

两个坑值得记下来:

坑 1:orgId 不能猜。深市老股是 gssz0000858,但洋河/今世缘/迎驾贡/口子窖这类
      是 99000xxxxx。手猜 gssz0002304 会稳定返回空列表,且**不报错**。
      所以一律走 topSearch 反查,结果缓存到 data/orgids.json。

坑 2:缺 UA/Referer 时巨潮返回的是 HTML 错误页,但 HTTP 状态码仍是 200。
      不检查 Content-Type 直接 .json() 会得到一个看不懂的解析异常。
"""

import argparse
import hashlib
import html
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

log = config.get_logger("01_download")

ORGID_CACHE = config.DATA_DIR / "orgids.json"
MANIFEST = config.DATA_DIR / "manifest.jsonl"

NOISE_WORDS = ("摘要", "英文", "English", "更正", "补充", "取消", "问询", "回复", "公告")


def sanitize(name: str, max_len: int = 80) -> str:
    """清掉 Windows 非法字符。中文标题本身是干净的,主要是防 URL 日期段之类。"""
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", name).strip(" _.")
    return name[:max_len] or "unnamed"


def _norm_title(t: str) -> str:
    """标题归一:去 HTML 标签、去实体、去所有空白。isHLtitle 会塞 <em> 进来。"""
    t = re.sub(r"<[^>]+>", "", t or "")
    t = html.unescape(t)
    return re.sub(r"\s+", "", t)


def _post(url: str, data: dict, retries: int = 3) -> dict:
    last = None
    for i in range(retries):
        try:
            r = requests.post(url, data=data, headers=config.CNINFO_HEADERS, timeout=30)
            ctype = r.headers.get("Content-Type", "")
            if "json" not in ctype.lower():
                # 典型症状:返回 HTML 错误页但状态码 200
                raise ValueError(f"非 JSON 响应 (Content-Type={ctype!r}), 前 120 字: {r.text[:120]!r}")
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"请求失败 {url}: {last}")


def resolve_orgid(session: requests.Session, code: str, cache: dict) -> str:
    """topSearch 反查 orgId。缓存,避免重复打接口。"""
    if code in cache and cache[code]:
        return cache[code]
    j = _post(config.CNINFO_TOP_SEARCH, {"keyWord": code, "maxNum": "10"})
    for item in j or []:
        if item.get("code") == code:
            cache[code] = item["orgId"]
            log.info("  orgId 反查: %s -> %s (%s)", code, item["orgId"], item.get("zwjc"))
            return item["orgId"]
    raise LookupError(f"topSearch 查不到 {code}")


def column_of(code: str) -> str:
    """交易所列名由代码前缀推导,不硬编码。6xx -> 上交所,其余 -> 深交所。"""
    return "sse" if code.startswith("6") else "szse"


def search_announcements(code: str, org_id: str, category: str, se_date: str) -> list[dict]:
    data = {
        "pageNum": "1",
        "pageSize": "30",
        "column": column_of(code),
        "tabName": "fulltext",
        "plate": "",
        "stock": f"{code},{org_id}",
        "searchkey": "",          # 保持为空:非空才会触发 <em> 注入
        "secid": "",
        "category": category,
        "trade": "",
        "seDate": se_date,
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    return _post(config.CNINFO_ANNOUNCE, data).get("announcements") or []


def pick_report(anns: list[dict], year: int, kind: str) -> dict | None:
    """从一堆公告里挑出真正那份报告。摘要/英文版/更正版必须滤掉。

    年份和「年度报告」之间的那个「年」字**不能写死**:实测水井坊 FY2023 的
    标题是「水井坊2023年度报告」,少一个「年」;多数公司写「2025年年度报告」。
    所以用 `年?` 容忍。
    """
    stem = rf"{year}年?年度报告" if kind == "annual" else rf"{year}年?半年度报告"
    pat = re.compile(stem)
    cands = []
    for a in anns:
        t = _norm_title(a.get("announcementTitle", ""))
        if not pat.search(t):
            continue
        if any(w in t for w in NOISE_WORDS):
            continue
        cands.append(a)
    if not cands:
        return None
    # 万一有重复,取最新发布的那份
    return max(cands, key=lambda a: a.get("announcementTime") or 0)


def download(session: requests.Session, url: str, dest: Path, force: bool = False) -> tuple[Path, int, str]:
    """下载 PDF,已存在且大小健康就跳过。返回 (路径, 字节数, sha256)。"""
    if dest.exists() and dest.stat().st_size >= config.DOWNLOAD_MIN_BYTES and not force:
        data = dest.read_bytes()
        return dest, len(data), hashlib.sha256(data).hexdigest()

    r = session.get(url, headers={"User-Agent": config.CNINFO_HEADERS["User-Agent"]},
                    timeout=120, stream=True)
    r.raise_for_status()
    buf = b"".join(r.iter_content(1 << 16))
    if len(buf) < config.DOWNLOAD_MIN_BYTES or not buf.startswith(b"%PDF"):
        raise ValueError(f"下载内容不像 PDF ({len(buf)} 字节, 开头 {buf[:8]!r})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(buf)
    return dest, len(buf), hashlib.sha256(buf).hexdigest()


def period_label_to_targets(periods: list[str]) -> list[tuple]:
    out = []
    for title_stem, category, se_date, label in config.PERIODS:
        if periods and label not in periods:
            continue
        year = int(re.search(r"(\d{4})", title_stem).group(1))
        # 必须先判"半年度报告"!"年度报告" 是 "半年度报告" 的子串
        # (半|年度报告),反过来判会把半年报错判成年报,12 家半年报会静默全丢。
        kind = "semi" if "半年度报告" in title_stem else "annual"
        out.append((category, se_date, label, year, kind))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="下载白酒公司财报 PDF")
    ap.add_argument("--only", default="", help="只跑这几家,逗号分隔代码,如 600519,000858")
    ap.add_argument("--periods", default="", help="只跑这几期,逗号分隔标签,如 FY2025,H1-2026")
    ap.add_argument("--force", action="store_true", help="强制重新下载")
    args = ap.parse_args()

    companies = json.loads(config.COMPANIES_FILE.read_text(encoding="utf-8"))["companies"]
    if args.only:
        want = {c.strip() for c in args.only.split(",") if c.strip()}
        companies = [c for c in companies if c["code"] in want]
    targets = period_label_to_targets([p.strip() for p in args.periods.split(",") if p.strip()])

    cache = json.loads(ORGID_CACHE.read_text(encoding="utf-8")) if ORGID_CACHE.exists() else {}
    session = requests.Session()
    records, ok, missing = [], 0, []

    for ci, comp in enumerate(companies, 1):
        code, name = comp["code"], comp["name"]
        log.info("[%d/%d] %s %s", ci, len(companies), code, name)
        try:
            org_id = resolve_orgid(session, code, cache)
        except Exception as e:  # noqa: BLE001
            log.error("  orgId 反查失败: %s", e)
            missing.append((code, name, "ALL", str(e)))
            continue

        for category, se_date, label, year, kind in targets:
            try:
                anns = search_announcements(code, org_id, category, se_date)
                hit = pick_report(anns, year, kind)
                if not hit:
                    log.warning("  %s 未找到 (%d 条公告里没匹配上)", label, len(anns))
                    missing.append((code, name, label, f"{len(anns)} 条公告无匹配"))
                    continue
                title = _norm_title(hit["announcementTitle"])
                url = config.CNINFO_STATIC + hit["adjunctUrl"]
                dest = config.RAW_DIR / f"{code}_{name}" / f"{label}_{sanitize(title)}.pdf"
                path, nbytes, sha = download(session, url, dest, force=args.force)
                records.append({
                    "code": code, "name": name, "short": comp["short"], "org_id": org_id,
                    "period": label, "period_year": year, "kind": kind, "title": title,
                    "url": url, "path": str(path.relative_to(config.BASE_DIR)),
                    "bytes": nbytes, "sha256": sha, "announcement_time": hit.get("announcementTime"),
                })
                ok += 1
                log.info("  %-8s %6.1fMB  %s", label, nbytes / 1e6, title[:46])
            except Exception as e:  # noqa: BLE001
                log.error("  %s 失败: %s", label, e)
                missing.append((code, name, label, str(e)))
            time.sleep(config.DOWNLOAD_SLEEP)

    ORGID_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    with MANIFEST.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(config.json_line(r) + "\n")

    log.info("=" * 56)
    log.info("下载完成: %d 份 -> %s", ok, config.RAW_DIR)
    log.info("清单: %s", MANIFEST)
    if missing:
        log.warning("缺失 %d 项:", len(missing))
        for m in missing:
            log.warning("  %s %s %s -> %s", *m)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
