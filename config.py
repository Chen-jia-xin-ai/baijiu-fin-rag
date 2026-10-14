# -*- coding: utf-8 -*-
"""全局配置 + 零依赖 .env 加载。

沿用 auto-agent/config.py 的两个约定:
  1. 本机私密信息(API key)只从环境变量或 .env 读,绝不写进代码,也绝不入库;
  2. Windows 控制台默认 GBK,中文日志会炸,统一切到 utf-8。
"""

import json
import logging
import os
import sys
from pathlib import Path

# ---------- 路径 ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"          # 下载的 PDF
PAGES_DIR = DATA_DIR / "pages"      # 逐页提取结果 (jsonl)
CHUNKS_DIR = DATA_DIR / "chunks"    # 切块结果 (jsonl)
TABLES_DIR = DATA_DIR / "tables"    # 表格结构化行 (jsonl),供 pandas 兜底
INDEX_DIR = DATA_DIR / "index"      # 向量 + BM25 索引
EVAL_DIR = BASE_DIR / "eval"
DOCS_DIR = BASE_DIR / "docs"
LOG_DIR = BASE_DIR / "logs"
ENV_FILE = BASE_DIR / ".env"

for _d in (RAW_DIR, PAGES_DIR, CHUNKS_DIR, TABLES_DIR, INDEX_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Windows 控制台默认 GBK,中文日志会炸。统一切到 utf-8。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def load_env(path: Path = ENV_FILE) -> None:
    """极简 .env 解析:KEY=VALUE,支持 # 注释与引号。已存在的环境变量优先。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


load_env()

# HuggingFace 直连在本机被墙(实测 000),hf-mirror 直连可用(200)。
# 必须在 import fastembed / huggingface_hub 之前设好。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---------- 语料 ----------
# 每家公司取这几期。年报 3 期是为了支撑"跨报告期"的题;半年报是低成本的对照。
PERIODS = [
    ("2023年年度报告", "category_ndbg_szsh", "2024-01-01~2024-12-31", "FY2023"),
    ("2024年年度报告", "category_ndbg_szsh", "2025-01-01~2025-12-31", "FY2024"),
    ("2025年年度报告", "category_ndbg_szsh", "2026-01-01~2026-12-31", "FY2025"),
    ("2026年半年度报告", "category_bndbg_szsh", "2026-06-01~2026-12-31", "H1-2026"),
]

COMPANIES_FILE = BASE_DIR / "companies.json"

# ---------- 巨潮 ----------
CNINFO_TOP_SEARCH = "http://www.cninfo.com.cn/new/information/topSearch/query"
CNINFO_ANNOUNCE = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC = "http://static.cninfo.com.cn/"
CNINFO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "Referer": "http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice",
}
DOWNLOAD_MIN_BYTES = 100 * 1024   # 小于 100KB 一律当下载失败
DOWNLOAD_SLEEP = 0.8              # 别把人家网站打疼

# ---------- 切块 ----------
# bge-small-zh-v1.5 是 512 token 截断,中文约 1 字 1 token,
# 减去上下文前缀(~30 字)后正文上限压到 CHUNK_MAX_CHARS。
CHUNK_MAX_CHARS = 440
CHUNK_OVERLAP_CHARS = 60
TABLE_MAX_CHARS = 700             # 单块放得下的表格线性化文本上限,超了按行切

# ---------- 检索 ----------
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"   # 512 维,fastembed onnx_embedding 内置
EMBED_BATCH = 64
RETRIEVE_PER_PATH = 20            # 向量、BM25 各取 top-20 再融合
RRF_K = 60
TOP_K_DEFAULT = 5
BM25_WEIGHT = 1.0                 # 本语料词法不弱于向量,不额外压低
VECTOR_WEIGHT = 1.0

# ---------- 生成 ----------
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_URL = f"{DEEPSEEK_BASE_URL}/chat/completions"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
# 优先专用 key;没有就回退复用本机 Claude Code 那个 Anthropic 兼容 token。
DEEPSEEK_API_KEY = (
    os.getenv("DEEPSEEK_API_KEY", "").strip()
    or os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip()
)
LLM_ENABLED = bool(DEEPSEEK_API_KEY)
LLM_TIMEOUT = 90
LLM_CACHE_FILE = DATA_DIR / "llm_cache.json"

# ---------- 问答页 ----------
APP_PORT = int(os.getenv("APP_PORT", "8000"))

# ---------- JSONL 读写 ----------

def read_jsonl(path) -> list:
    """按 `\\n` 切行读 JSONL。

    **不要用 `str.splitlines()`** —— 财报正文里会出现 U+2028 / U+2029 行分隔符
    (PDF 排版遗留),`splitlines()` 会把它们当换行,于是一条记录被切成两半,
    报「Unterminated string」。实测 山西汾酒 FY2023 和 水井坊 FY2023 都中招。
    """
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").split("\n"):
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def json_line(obj) -> str:
    """一行 JSON。U+2028 / U+2029 显式转义 —— 它们在 JSON 里合法,但会被
    不少工具当成换行,一条记录就裂成两条。写出去的时候堵死这个口子。"""
    s = json.dumps(obj, ensure_ascii=False)
    return s.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def write_jsonl(path, rows) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json_line(r) + "\n")


# ---------- 日志 ----------


def get_logger(name: str) -> logging.Logger:
    """同时写控制台和 logs/<name>_YYYYMMDD.log。"""
    from datetime import date

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(
        LOG_DIR / f"{name}_{date.today():%Y%m%d}.log", encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger
