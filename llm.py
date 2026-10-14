# -*- coding: utf-8 -*-
"""DeepSeek 调用 + 本地缓存。网页与评测共用。

两条纪律:

1. **模型不许自己写公司名和页码。** 引用编号 `[1][2]` 由调用方(服务端)
   按检索结果分配,模型只准引用编号。让它自由发挥就一定会编出处,
   而「答案带出处」正是这次作业的考点 —— 出处必须是检索给的,不能是生成的。

2. **全部响应落盘缓存。** 缓存键 = 问题 + 命中块 id + prompt 版本。
   截图和评测因此可重复、免费,而且改了 prompt 会自动失效重算。
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

PROMPT_VERSION = "v3"

ANSWER_SYSTEM = """你是一家白酒行业研究员的财报助手。你只根据给出的【材料】回答,不使用任何记忆中的数字。

硬性要求:
1. 每句涉及具体数字或事实的话后面必须跟引用编号,如 [1] 或 [1][3]。没有编号的断言视为无效。
2. 只允许引用材料里出现过的数字,禁止推算、禁止四舍五入改值、禁止凭常识补全。
3. 注意「合并」与「母公司」两套口径数字不同,引用时必须写明是哪个口径(材料前缀里有)。
4. 注意单位。材料里写「单位:万元」就是万元,不要按元报。
5. 材料不足以回答时,明确说「现有材料不足以回答」,并说明缺什么。不要猜。
6. 回答简洁,先给结论再给依据。涉及多期或多公司比较时用短表格或列表。"""

JUDGE_SYSTEM = """你是严格的阅卷人,给一个回答打分。只看评分细则,不接受「意思差不多」。

输出严格 JSON(不要 markdown 代码块):
{"verdict": "CORRECT" | "PARTIAL" | "WRONG" | "REFUSED", "reason": "一句话理由"}
CORRECT = 关键事实与金标一致(数值在容差内、口径正确、单位正确);
PARTIAL = 部分正确或缺一项;
WRONG = 与金标冲突;
REFUSED = 回答拒绝作答/答非所问,而材料其实足以作答。"""


def has_key() -> bool:
    return bool(config.DEEPSEEK_API_KEY)


def _cache_key(kind: str, payload: dict) -> str:
    raw = json.dumps([PROMPT_VERSION, kind, payload], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class _Cache:
    def __init__(self, path: Path):
        self.path = path
        self.data = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self.data = {}
        self.dirty = False
        self.hits = 0

    def get(self, key: str):
        v = self.data.get(key)
        if v is not None:
            self.hits += 1
        return v

    def put(self, key: str, value) -> None:
        self.data[key] = value
        self.dirty = True
        self.flush()

    def flush(self) -> None:
        if not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        self.dirty = False


_cache: _Cache | None = None


def cache() -> _Cache:
    global _cache
    if _cache is None:
        _cache = _Cache(config.LLM_CACHE_FILE)
    return _cache


def _chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 1200,
          retries: int = 3) -> str:
    last = None
    for i in range(retries):
        try:
            r = requests.post(
                config.DEEPSEEK_URL,
                headers={"Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": config.DEEPSEEK_MODEL, "messages": messages,
                      "temperature": temperature, "max_tokens": max_tokens, "stream": False},
                timeout=120,
            )
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"DeepSeek 调用失败: {last}")


# ---------- 材料拼装 ----------

def build_context(hits: list[dict], snippet: int = 1800) -> str:
    """把命中块编号后拼成材料。编号就是引用编号,由检索名次决定。"""
    blocks = []
    for i, h in enumerate(hits, 1):
        body = h["text"]
        if len(body) > snippet:
            body = body[:snippet] + " …(截断)"
        blocks.append(f"[{i}] 出处:{h['citation']}\n{body}")
    return "\n\n".join(blocks)


# ---------- 对外 ----------

def answer(question: str, hits: list[dict]) -> dict:
    """基于检索块作答。无 key 时优雅降级,不抛异常。

    返回 {answer, model, cached, note}
    """
    if not hits:
        return {"answer": None, "model": None, "cached": False, "note": "没有召回到任何材料。"}
    if not has_key():
        return {"answer": None, "model": None, "cached": False,
                "note": "未检测到 DEEPSEEK_API_KEY,已跳过生成 —— 下面是检索到的原始材料。"}

    key = _cache_key("answer", {"q": question, "chunks": [h["chunk_id"] for h in hits]})
    c = cache()
    got = c.get(key)
    if got is not None:
        return {**got, "cached": True}

    user = (f"【材料】\n{build_context(hits)}\n\n"
            f"【问题】{question}\n\n"
            f"请依据材料作答,每句事实后标注引用编号。")
    try:
        text = _chat([{"role": "system", "content": ANSWER_SYSTEM},
                      {"role": "user", "content": user}])
    except Exception as e:  # noqa: BLE001
        return {"answer": None, "model": None, "cached": False, "note": f"生成失败:{e}"}

    out = {"answer": text, "model": config.DEEPSEEK_MODEL, "note": ""}
    c.put(key, out)
    return {**out, "cached": False}


def judge(question: str, gold: str, model_answer: str) -> dict:
    """LLM 阅卷。注意:同一家模型评自己写的答案,存在自我偏好 —— 报告里必须写明。"""
    if not has_key():
        return {"verdict": "SKIPPED", "reason": "无 API key"}
    key = _cache_key("judge", {"q": question, "gold": gold, "ans": model_answer})
    c = cache()
    got = c.get(key)
    if got is not None:
        return {**got, "cached": True}

    user = (f"【问题】{question}\n\n【金标答案】{gold}\n\n【待评回答】{model_answer}\n\n"
            f"请按细则打分,输出 JSON。")
    try:
        raw = _chat([{"role": "system", "content": JUDGE_SYSTEM},
                     {"role": "user", "content": user}], max_tokens=300)
    except Exception as e:  # noqa: BLE001
        return {"verdict": "ERROR", "reason": str(e)}

    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        out = json.loads(text)
    except Exception:  # noqa: BLE001
        out = {"verdict": "PARSE_ERROR", "reason": raw[:160]}
    out.setdefault("verdict", "PARSE_ERROR")
    out.setdefault("reason", "")
    c.put(key, out)
    return {**out, "cached": False}
