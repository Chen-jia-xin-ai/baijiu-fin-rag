# -*- coding: utf-8 -*-
"""检索层:向量 + BM25 混合检索,跨公司问题分路检索。

三件事值得说清:

1. **索引的是 `embed_text` 而不是 `text`。** 每个块前面那行自描述前缀
   (公司 / 报告期 / 章节 / 页码 / 合并-母公司)一起进 BM25 词表和向量。
   否则一个写着「期末余额 1,234」的块在 12 家公司里向量上无法区分,
   所有全景题都会塌向语料占比最大的那家。

2. **跨公司问题必须分路检索。** 融合出的单一 top-5 会被其中一家霸榜:
   「汾酒和泸州老窖谁的毛利率高」在全局 top-5 里很可能全是汾酒,
   因为两家的段落措辞高度相似。这里检测查询里提到的公司,逐家各检索一次
   再融合 —— 这一个改动的收益大于任何重排器。

3. **BM25 用 `BM25Plus` 而不是 `BM25Okapi`。** Okapi 会给高 df 的词负 idf,
   而「公司名」在本语料里恰恰是高 df 词,负 idf 会让它起反作用。

向量索引不引入 faiss:三万多块 × 512 维,归一化后直接矩阵点积是毫秒级,
少一个重依赖就少一处装不上的地方。
"""

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# 财务长词必须进自定义词典,否则 jieba 会把最有价值的词切碎
# (「归属于上市公司股东的净利润」会被切成五六段,BM25 上就废了)
FINANCE_TERMS = [
    "归属于上市公司股东的净利润", "归属于上市公司股东的扣除非经常性损益的净利润",
    "经营活动产生的现金流量净额", "投资活动产生的现金流量净额", "筹资活动产生的现金流量净额",
    "加权平均净资产收益率", "基本每股收益", "稀释每股收益", "扣除非经常性损益",
    "合同负债", "预收款项", "应收账款", "应收票据", "存货跌价准备", "递延所得税资产",
    "销售毛利率", "销售净利率", "销售费用率", "管理费用率", "毛利率", "净利率",
    "营业总收入", "营业收入", "营业成本", "税金及附加", "销售费用", "管理费用", "研发费用",
    "财务费用", "资产减值损失", "信用减值损失", "营业利润", "利润总额", "净利润",
    "流动资产", "非流动资产", "流动负债", "非流动负债", "所有者权益", "股东权益",
    "资产负债率", "货币资金", "交易性金融资产", "应付账款", "应付职工薪酬", "应交税费",
    "经销商", "合同资产", "使用权资产", "租赁负债", "少数股东权益", "母公司", "合并报表",
    "分产品", "分地区", "分销售模式", "前五名客户", "前五名供应商", "产能利用率",
    "高档酒", "中档酒", "普通酒", "系列酒", "基酒", "成品酒", "窖池", "产能",
]

_terms_loaded = False


def _load_terms() -> None:
    global _terms_loaded
    if _terms_loaded:
        return
    import jieba
    for t in FINANCE_TERMS:
        jieba.add_word(t, freq=100000)
    try:
        comps = json.loads(config.COMPANIES_FILE.read_text(encoding="utf-8"))["companies"]
        for c in comps:
            for alias in [c["name"], c["short"], *c.get("aliases", [])]:
                if alias:
                    jieba.add_word(alias, freq=100000)
    except Exception:  # noqa: BLE001
        pass
    _terms_loaded = True


def tokenize(text: str) -> list[str]:
    _load_terms()
    import jieba
    return [t for t in jieba.lcut_for_search(text.lower()) if t.strip()]


def load_companies() -> list[dict]:
    return json.loads(config.COMPANIES_FILE.read_text(encoding="utf-8"))["companies"]


def mentioned_companies(query: str, companies: list[dict] | None = None) -> list[str]:
    """查询里提到了哪几家公司(按别名匹配)。用于跨公司问题的分路检索。"""
    companies = companies or load_companies()
    hits = []
    for c in companies:
        for alias in sorted({c["name"], c["short"], *c.get("aliases", [])}, key=len, reverse=True):
            if alias and alias in query:
                hits.append(c["code"])
                break
    return hits


# ---------- 查询里解析得出的结构化字段 ----------
#
# 第一轮评测最扎眼的结论:**失效主因不是「跨公司塌陷」,是「报告期没被当成字段」。**
# 「2025年」确实写进了自描述前缀、也确实进了 BM25 词表,但它只是**一段文本**。
# 一份 FY2023 年报里逐字相同的「可能面对的风险」段落,在向量上同样像答案 ——
# Q1 的第 3、4 名就是 FY2023 第 2 页和第 21 页,第 5 名是 H1-2026 第 12 页,
# 全是**正确的内容、错误的年份**。Q5 分路检索后 5 条里 4 条是汾酒的 FY2023/H1-2026,
# 泸州老窖一条都没有。BM25 更糟:它把「2025」和「2023」都当成普通词,
# 年份这种高 df 数字在 BM25Plus 里几乎没有区分力。
#
# 结论:年月要从「文本」升级成「可过滤字段」。这就是「该建库」的字面含义 ——
# 不是把事实埋进向量,是给事实**结构化字段**。
PERIOD_YEAR_RE = re.compile(r"(20\d\d)\s*年")
HALF_RE = re.compile(r"半年报|半年度|中期|中报")


def periods_from_query(query: str, known: set[str] | None = None) -> set[str]:
    """从问题里解析报告期。解析不出就返回空集 —— 宁可不收紧,不可乱收紧。"""
    years = set(PERIOD_YEAR_RE.findall(query))
    if not years:
        return set()
    half = bool(HALF_RE.search(query))
    out = {f"H1-{y}" if half else f"FY{y}" for y in years}
    return out & known if known else out


JUNK_RE = re.compile(r"[A-Za-z0-9+/=._-]{24,}")


def is_junk_chunk(c: dict) -> bool:
    """二维码 / 验证码 / 网址正文,以及目录页的点线。

    两样都不是「检索不到」,而是「检索到了更糟」:
      - 今世缘 H1-2026 第 16 页整页是信息披露二维码,那两串验证码字数长、
        字符分布古怪,向量上反而**很像**一张密密麻麻的数字表,第一轮直接拿到
        BM25 220 分冲进 Q4 的 top-5(第 2、4 名)。
      - 目录页(「第一节 重要提示、目录和释义 ............ 3」)会被
        `find_tables` 当成一张表抓下来,列名退化成「列1 列2 列3」,内容全是点线。
    """
    t = c.get("text", "")
    if JUNK_RE.search(t):
        return True
    return c.get("chunk_type") == "table" and t.count("....") >= 2


class Retriever:
    def __init__(self, chunks, embeddings, bm25, tokens, companies,
                 rows=None, rows_emb=None, row_bm25=None):
        self.chunks = chunks
        self.embeddings = embeddings            # (N, D) float32, L2 归一化
        self.bm25 = bm25
        self.companies = companies
        # 行级块:和页级表格块同形,只是每个块只装一行。见 scripts/08_rows.py
        self.rows = rows or []
        self.rows_emb = rows_emb
        self.row_bm25 = row_bm25
        self._by_code = self._index_by(chunks, "code")
        self._by_period = self._index_by(chunks, "period")
        self._row_by_code = self._index_by(self.rows, "code")
        self._row_by_period = self._index_by(self.rows, "period")
        # 噪声块一律出局。这事必须做在检索层而不是切块层:
        # 切块层丢不掉「这一块是二维码」这个判断,但检索层知道它排在前面。
        self.ok = np.array([not is_junk_chunk(c) for c in chunks], dtype=bool)
        self.n_junk = int((~self.ok).sum())
        self._model = None

    @staticmethod
    def _index_by(recs, field: str) -> dict:
        d: dict[str, list[int]] = {}
        for i, c in enumerate(recs):
            d.setdefault(c[field], []).append(i)
        return {k: np.array(v, dtype=np.int64) for k, v in d.items()}

    @staticmethod
    def _select(n, by_code, by_period, codes=None, periods=None):
        """把「公司 / 报告期」过滤条件翻成一个下标数组;两者都没给就返回 None。

        None 和「空数组」是两回事:None = 不限制(走全量矩阵点积),
        空数组 = 限制之后什么都不剩(直接返回空,别浪费一次点积)。
        """
        if not codes and not periods:
            return None
        mask = np.zeros(n, dtype=bool)
        if codes:
            kept = [by_code[c] for c in codes if c in by_code]
            if not kept:
                return np.array([], dtype=np.int64)
            for a in kept:
                mask[a] = True
        else:
            mask[:] = True
        if periods:
            pmask = np.zeros(n, dtype=bool)
            for p in periods:
                if p in by_period:
                    pmask[by_period[p]] = True
            mask &= pmask
        return np.flatnonzero(mask)

    # ---------- 模型 ----------
    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=config.EMBED_MODEL)
        return self._model

    def embed_query(self, query: str) -> np.ndarray:
        try:
            vec = next(iter(self.model.query_embed([query])))
        except Exception:  # noqa: BLE001
            vec = next(iter(self.model.embed([query])))
        v = np.asarray(vec, dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n else v

    # ---------- 四条召回路径 ----------
    def _vector_top(self, query: str, depth: int, codes=None, periods=None
                    ) -> list[tuple[int, float]]:
        q = self.embed_query(query)
        sel = self._select(len(self.chunks), self._by_code, self._by_period, codes, periods)
        if sel is None:
            sims = np.where(self.ok, self.embeddings @ q, -np.inf)
            order = np.argsort(-sims)[:depth]
            return [(int(i), float(sims[i])) for i in order if np.isfinite(sims[i])]
        if sel.size == 0:
            return []
        sel = sel[self.ok[sel]]          # 噪声块出局
        if sel.size == 0:
            return []
        sims = self.embeddings[sel] @ q
        order = np.argsort(-sims)[:depth]
        return [(int(sel[i]), float(sims[i])) for i in order]

    def _bm25_top(self, query: str, depth: int, codes=None, periods=None
                  ) -> list[tuple[int, float]]:
        scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)
        sel = self._select(len(self.chunks), self._by_code, self._by_period, codes, periods)
        if sel is None:
            scores = np.where(self.ok, scores, -np.inf)
            order = np.argsort(-scores)[:depth]
            return [(int(i), float(scores[i])) for i in order if np.isfinite(scores[i])]
        if sel.size == 0:
            return []
        sel = sel[self.ok[sel]]
        if sel.size == 0:
            return []
        scores = scores[sel]
        order = np.argsort(-scores)[:depth]
        return [(int(sel[i]), float(scores[i])) for i in order]

    def _row_vector_top(self, query: str, depth: int, codes=None, periods=None
                        ) -> list[tuple[int, float]]:
        if self.rows_emb is None:
            return []
        q = self.embed_query(query)
        sel = self._select(len(self.rows), self._row_by_code, self._row_by_period, codes, periods)
        if sel is None:
            sims = self.rows_emb @ q
            order = np.argsort(-sims)[:depth]
            return [(int(i), float(sims[i])) for i in order]
        if sel.size == 0:
            return []
        sims = self.rows_emb[sel] @ q
        order = np.argsort(-sims)[:depth]
        return [(int(sel[i]), float(sims[i])) for i in order]

    def _row_bm25_top(self, query: str, depth: int, codes=None, periods=None
                      ) -> list[tuple[int, float]]:
        if self.row_bm25 is None:
            return []
        scores = np.asarray(self.row_bm25.get_scores(tokenize(query)), dtype=np.float32)
        sel = self._select(len(self.rows), self._row_by_code, self._row_by_period, codes, periods)
        if sel is None:
            order = np.argsort(-scores)[:depth]
            return [(int(i), float(scores[i])) for i in order]
        if sel.size == 0:
            return []
        scores = scores[sel]
        order = np.argsort(-scores)[:depth]
        return [(int(sel[i]), float(scores[i])) for i in order]

    def _one_path(self, query: str, path: str, depth: int, codes, periods):
        """跑一条路径,统一成 `[(kind, idx, 向量分, BM25 分)]`,按原始分降序。"""
        if path == "vector":
            return [("c", i, s, 0.0) for i, s in self._vector_top(query, depth, codes, periods)]
        if path == "bm25":
            return [("c", i, 0.0, s) for i, s in self._bm25_top(query, depth, codes, periods)]
        if path == "row_vec":
            return [("r", i, s, 0.0) for i, s in self._row_vector_top(query, depth, codes, periods)]
        if path == "row_bm25":
            return [("r", i, 0.0, s) for i, s in self._row_bm25_top(query, depth, codes, periods)]
        return []

    def _fuse(self, query: str, paths: list[str], depth: int, codes=None, periods=None
              ) -> list[dict]:
        """RRF 融合若干条路径。

        单路径时不做融合,直接把原始分当融合分 —— 这样「仅向量 / 仅 BM25」
        两个对照模式的报分口径和第一轮一致,前后可比。
        """
        if len(paths) == 1:
            hits = self._one_path(query, paths[0], depth, codes, periods)
            return [self._make_hit(k, i, vs, bs, vs or bs, rank)
                    for rank, (k, i, vs, bs) in enumerate(hits, 1)]
        fused: dict[tuple, dict] = {}
        for path in paths:
            for rank, (k, i, vs, bs) in enumerate(
                    self._one_path(query, path, depth, codes, periods), 1):
                e = fused.setdefault((k, i), {"f": 0.0, "vs": 0.0, "bs": 0.0})
                e["f"] += 1.0 / (config.RRF_K + rank)
                if vs:
                    e["vs"] = vs
                if bs:
                    e["bs"] = bs
        order = sorted(fused.items(), key=lambda kv: -kv[1]["f"])[:depth]
        return [self._make_hit(k, i, e["vs"], e["bs"], e["f"], rank)
                for rank, ((k, i), e) in enumerate(order, 1)]

    def _make_hit(self, kind: str, idx: int, vscore: float, bscore: float,
                  fscore: float, rank: int) -> dict:
        c = self.chunks[idx] if kind == "c" else self.rows[idx]
        return {
            "idx": idx, "kind": kind, "rank": rank,
            "score": round(fscore, 6),
            "score_vector": round(vscore, 4), "score_bm25": round(bscore, 4),
            "chunk_id": c["chunk_id"], "code": c["code"], "name": c["name"],
            "period": c["period"], "chunk_type": c["chunk_type"], "page": c["page"],
            "section": c["section"], "scope": c.get("scope", ""), "unit": c.get("unit", ""),
            "caption": c.get("caption", ""),
            "text": c["text"], "embed_text": c["embed_text"],
        }

    # ---------- 对外 ----------
    # 模式 -> 路径组合。`hybrid-row` 就是把行级索引当第三、第四条路径并进来:
    # 页级块给上下文,行级块给**精确的那一行**。两者不是替代关系。
    MODES = {
        "vector":     ["vector"],
        "bm25":       ["bm25"],
        "row":        ["row_vec", "row_bm25"],
        "hybrid":     ["vector", "bm25"],
        "hybrid-row": ["vector", "bm25", "row_vec", "row_bm25"],
    }

    def _route_multi(self, query, paths, depth, codes, top_k, periods):
        """逐家分路检索,再**按公司轮流占位**合并。

        第一轮的分路是「各取 15 条再按融合分排序」—— 这等于没分路:
        汾酒的段落在融合分上普遍高于泸州老窖,前 5 名照样被一家占满
        (Q5 实测 4/5 是汾酒、泸州老窖 0 条,题目直接答不成立)。
        改成轮流占位:每家先拿走 `ceil(top_k / 家数)` 个名额,
        谁分高谁先填,但**没有任何一家能把别人挤光**。
        最后保留轮流占位的顺序,不再按分重排 —— 存在性比分数重要。
        """
        per = [self._fuse(query, paths, depth, [c], periods) for c in codes]
        cursors = [0] * len(per)
        taken: set[tuple] = set()
        out: list[dict] = []

        def take_one(i: int) -> bool:
            """第 i 家往后找一条没被别家拿走的;拿到返回 True。"""
            while cursors[i] < len(per[i]):
                h = per[i][cursors[i]]
                cursors[i] += 1
                key = (h["kind"], h["idx"])
                if key not in taken:
                    taken.add(key)
                    out.append(h)
                    return True
            return False

        quota = max(1, -(-top_k // len(codes)))
        for _ in range(quota):                            # 第一轮:每家先拿满配额
            for i in range(len(per)):
                if len(out) >= top_k:
                    break
                take_one(i)
        while len(out) < top_k:                           # 第二轮:有余量就继续轮转
            progressed = False
            for i in range(len(per)):
                if len(out) >= top_k:
                    break
                if take_one(i):
                    progressed = True
            if not progressed:
                break
        return out[:top_k]

    def search(self, query: str, top_k: int = None, mode: str = "hybrid",
               codes: list[str] | None = None, route: bool = True,
               periods: set[str] | None = None, hard_filter: bool = False) -> dict:
        """返回 {hits, routed_companies, mode, filters}。

        hard_filter=True 时,把查询里解析出的**公司和报告期当成硬条件**
        (AND 进候选集),而不是只当检索词。这是第一轮评测指向的那条修法:
        「2025 年」写在文本里和「2025 年」是个过滤字段,是两回事。
        """
        top_k = top_k or config.TOP_K_DEFAULT
        depth = config.RETRIEVE_PER_PATH
        paths = self.MODES.get(mode, self.MODES["hybrid"])

        if hard_filter and not codes:
            codes = mentioned_companies(query, self.companies) or None
        if hard_filter and not periods:
            periods = periods_from_query(query, {c["period"] for c in self.chunks}) or None

        routed = []
        if codes and len(codes) >= 2:
            routed = list(codes)
            hits = self._route_multi(query, paths, depth, codes, top_k, periods)
        elif codes:
            hits = self._fuse(query, paths, depth, codes, periods)[:top_k]
        else:
            mentioned = mentioned_companies(query, self.companies) if route else []
            if len(mentioned) >= 2:
                routed = mentioned
                hits = self._route_multi(query, paths, depth, mentioned, top_k, periods)
            else:
                hits = self._fuse(query, paths, depth, None, periods)[:top_k]
        for i, h in enumerate(hits, 1):
            h["rank"] = i
        # 合并成一个可读的出处串,页面和评测都用它
        for h in hits:
            bits = [f"{h['name']} {h['period']}", f"第{h['page']}页"]
            if h["section"]:
                bits.append(h["section"])
            if h["scope"]:
                bits.append(h["scope"])
            if h["chunk_type"] == "row":
                bits.append("表格(行级" + (f":{h['caption']}" if h["caption"] else "") + ")")
            elif h["chunk_type"] == "table":
                bits.append("表格" + (f"({h['caption']})" if h["caption"] else ""))
            h["citation"] = " · ".join(bits)
        return {"hits": hits, "routed_companies": routed, "mode": mode, "query": query,
                "filters": {"codes": codes, "periods": sorted(periods) if periods else None}}

    def get_chunk(self, chunk_id: str) -> dict | None:
        for c in self.chunks:
            if c["chunk_id"] == chunk_id:
                return c
        return None


# ---------- 载入 / 构建 ----------

def load(cache_dir: Path | None = None) -> Retriever:
    """从 data/index 载入已建好的索引(秒级)。

    行级索引是**可选**的:没跑过 `scripts/08_rows.py` 也能起来,
    只是 `row` / `hybrid-row` 两个模式会退化成空结果。仓库是公开的,
    评审没跑全链路时不该看到 traceback。
    """
    d = cache_dir or config.INDEX_DIR
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    chunks = config.read_jsonl(d / "chunks.jsonl")
    embeddings = np.load(d / "embeddings.npy")
    with (d / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)
    rows = rows_emb = row_bm25 = None
    if (d / "rows_bm25.pkl").exists():
        rows = config.read_jsonl(d / "rows.jsonl")
        with (d / "rows_bm25.pkl").open("rb") as f:
            row_bm25 = pickle.load(f)
        if (d / "rows_emb.npy").exists():
            rows_emb = np.load(d / "rows_emb.npy")
    return Retriever(chunks, embeddings, bm25, meta["n_tokens"], load_companies(),
                     rows=rows, rows_emb=rows_emb, row_bm25=row_bm25)


def build_index(force: bool = False) -> Retriever:
    """读 chunks.jsonl -> 建 BM25 + 向量 -> 落盘 data/index。"""
    from rank_bm25 import BM25Plus

    src = config.CHUNKS_DIR / "chunks.jsonl"
    chunks = config.read_jsonl(src)
    print(f"载入 {len(chunks)} 块", flush=True)

    tok_path = config.INDEX_DIR / "tokens.pkl"
    if tok_path.exists() and not force:
        with tok_path.open("rb") as f:
            tokens = pickle.load(f)
        print(f"复用分词缓存 {len(tokens)} 条", flush=True)
    else:
        tokens = [tokenize(c["embed_text"]) for c in chunks]
        with tok_path.open("wb") as f:
            pickle.dump(tokens, f)
        print("分词完成", flush=True)

    emb_path = config.INDEX_DIR / "embeddings.npy"
    if emb_path.exists() and not force:
        embeddings = np.load(emb_path)
        print(f"复用向量缓存 {embeddings.shape}", flush=True)
    else:
        from fastembed import TextEmbedding
        print(f"加载 {config.EMBED_MODEL} …", flush=True)
        model = TextEmbedding(model_name=config.EMBED_MODEL)
        texts = [c["embed_text"] for c in chunks]
        embeddings = np.zeros((len(texts), 512), dtype=np.float32)
        done = 0
        for vec in model.embed(texts, batch_size=config.EMBED_BATCH):
            embeddings[done] = vec
            done += 1
            if done % 2000 == 0:
                print(f"  已嵌入 {done}/{len(texts)}", flush=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings /= norms
        np.save(emb_path, embeddings)
        print(f"向量完成 {embeddings.shape}", flush=True)

    bm25 = BM25Plus(tokens)
    with (config.INDEX_DIR / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)

    with (config.INDEX_DIR / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(config.json_line(c) + "\n")
    (config.INDEX_DIR / "meta.json").write_text(json.dumps({
        "model": config.EMBED_MODEL, "n_chunks": len(chunks), "dim": int(embeddings.shape[1]),
        "n_tokens": len(tokens), "companies": [c["code"] for c in load_companies()],
        "periods": sorted({c["period"] for c in chunks}),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("索引已落盘 data/index/", flush=True)
    return Retriever(chunks, embeddings, bm25, tokens, load_companies())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="检索冒烟测试")
    ap.add_argument("query", nargs="?", help="查询串")
    ap.add_argument("--mode", default="hybrid",
                    choices=["hybrid", "vector", "bm25", "row", "hybrid-row"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-route", action="store_true", help="关掉跨公司分路检索")
    ap.add_argument("--filter", action="store_true", help="把查询里的公司/报告期当硬条件")
    ap.add_argument("--snippet", type=int, default=150)
    a = ap.parse_args()

    r = load()
    print(f"索引: {len(r.chunks):,} 页级块 + {len(r.rows):,} 行级块"
          f"(噪声块 {r.n_junk}) | {r.embeddings.shape} | 模式 {a.mode}\n")
    if not a.query:
        raise SystemExit(0)

    res = r.search(a.query, top_k=a.top_k, mode=a.mode,
                   route=not a.no_route, hard_filter=a.filter)
    if res["filters"]["codes"] or res["filters"]["periods"]:
        print(f"硬过滤: 公司={res['filters']['codes']} 报告期={res['filters']['periods']}")
    if res["routed_companies"]:
        print(f"检出多家公司,分路检索: {res['routed_companies']}")
    if res["filters"]["codes"] or res["routed_companies"]:
        print()
    for h in res["hits"]:
        print(f"[{h['rank']}] {h['score']:.5f} (vec {h['score_vector']} / bm25 {h['score_bm25']}) "
              f"{h['chunk_type']} {h['chunk_id']}")
        print(f"    {h['citation']}")
        print(f"    {h['text'][:a.snippet].replace(chr(10), ' ⏎ ')}\n")
