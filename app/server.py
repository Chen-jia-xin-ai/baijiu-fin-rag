# -*- coding: utf-8 -*-
"""问答页后端。

引用编号 `[1][2]` 由**服务端**分配:检索结果的名次就是编号,
模型只被允许引用编号,不许自己写公司名和页码。这样「答案带出处」
的出处一定是检索给的,而不是模型编的。

无 API key 时仍可用:返回 answer=null + 检索块 + 提示条,
让评审不开 key 也能看到检索质量(公开仓库里跑出 traceback 太难看)。

    python app/server.py            # http://127.0.0.1:8000
"""

import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import llm  # noqa: E402
import retriever  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="白酒财报问答")
R = None


def get_retriever():
    global R
    if R is None:
        R = retriever.load()
    return R


class AskIn(BaseModel):
    question: str
    top_k: int = 5
    mode: str = "hybrid-row"
    route: bool = True
    hard_filter: bool = True
    generate: bool = True


class SearchIn(BaseModel):
    question: str
    top_k: int = 10
    mode: str = "hybrid-row"
    route: bool = True
    hard_filter: bool = True


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/stats")
def stats():
    r = get_retriever()
    return {
        "n_chunks": len(r.chunks),
        "n_table_chunks": sum(1 for c in r.chunks if c["chunk_type"] == "table"),
        "n_rows": len(r.rows),
        "n_junk": r.n_junk,
        "companies": [{"code": c["code"], "name": c["name"], "short": c["short"]}
                      for c in r.companies],
        "periods": sorted({c["period"] for c in r.chunks}),
        "has_key": llm.has_key(),
        "model": config.DEEPSEEK_MODEL if llm.has_key() else None,
    }


@app.post("/api/search")
def search(body: SearchIn):
    r = get_retriever()
    return r.search(body.question, top_k=body.top_k, mode=body.mode,
                    route=body.route, hard_filter=body.hard_filter)


@app.post("/api/ask")
def ask(body: AskIn):
    r = get_retriever()
    if not body.question.strip():
        return JSONResponse({"error": "问题为空"}, status_code=400)
    res = r.search(body.question, top_k=body.top_k, mode=body.mode,
                   route=body.route, hard_filter=body.hard_filter)
    if body.generate:
        res["gen"] = llm.answer(body.question, res["hits"])
    else:
        res["gen"] = {"answer": None, "note": "未生成(仅检索)", "cached": False, "model": None}
    return res


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=config.APP_PORT, log_level="info")
