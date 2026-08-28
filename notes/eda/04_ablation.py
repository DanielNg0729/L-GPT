"""
EDA pass 4: ablation ladder, scored by the REAL evaluator.

Every variant below is run through evaluator.local_evaluator.evaluate() unmodified,
on all 200 public sessions. No evaluator file is touched; we only import it.

The ladder isolates the contribution of each mechanism:
    V0  starter BM25 (published reference: 0.1067)
    V1  + session state (accumulate every user message)
    V2  + ask_attribute='other' every turn (maximal probe)
    V3  + phrase queries instead of bag-of-terms OR
    V4  + constraint-aware scoring w/ AND->OR backoff ladder
    V5  + dense bi-encoder fused by RRF  (added in 05_dense.py)

Run:  PYTHONIOENCODING=utf-8 python notes/eda/04_ablation.py
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent as StarterAgent  # noqa: E402

CATALOG = ROOT / "data" / "catalog.jsonl"
PUBLIC = ROOT / "data" / "public_set.jsonl"
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

STOP = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i", "in", "is", "it",
    "me", "my", "of", "on", "or", "please", "some", "that", "the", "this", "to", "want", "with",
    "would", "you", "looking", "im", "still", "exploring", "key", "requirement", "what", "matters",
    "actually", "ignore", "earlier", "preference", "need", "dont", "have", "additional", "not",
    "quite", "right", "yet", "ask", "about", "one", "specific", "attribute", "those", "options",
    "judgment", "use", "your", "prefer", "different", "style", "prioritize", "target", "requirements",
}

# Templates the deterministic simulator emits. Parsed when present, but every
# variant ALSO falls back to raw-token extraction so paraphrasing cannot break it.
PAT_REQUIREMENT = re.compile(r"key requirement is:\s*(.+?)\.?$", re.I)
PAT_MATTERS = re.compile(r"what matters is:\s*(.+?)\.?$", re.I)
PAT_OVERRIDE = re.compile(r"what i need is:\s*(.+?)\.?$", re.I)
PAT_LOOKING = re.compile(r"looking for\s+(.+?)(?:[,.]|$)", re.I)
PAT_NOINFO = re.compile(r"don'?t have (?:an? )?(?:additional )?preference", re.I)


def toks(text: str) -> list[str]:
    """Content tokens, for bag-of-words OR queries."""
    return [t.lower() for t in TOKEN_RE.findall(text) if len(t) > 1 and t.lower() not in STOP]


def raw_toks(text: str) -> list[str]:
    """ALL tokens, in order, matching the FTS5 unicode61 tokenizer.

    Phrase queries assert token ADJACENCY against the indexed text. The index keeps
    stopwords, so a phrase built from stopword-filtered tokens can never match:
    "long torso camisole for extra coverage" indexes with 'for' in place, and the
    query "long torso camisole extra coverage" is then adjacency-false. This is a
    real bug that cost ~32 points of HR@10 in the first run of this ablation.
    """
    return [t.lower() for t in TOKEN_RE.findall(text)]


def phrase_of(text: str, cap: int = 12) -> str:
    t = raw_toks(text)[:cap]
    return '"' + " ".join(t) + '"' if t else ""


class Index:
    """Shared in-memory FTS5 index; built once, reused by every variant."""

    W = "bm25(p, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)"

    def __init__(self, path: Path):
        self.con = sqlite3.connect(":memory:", check_same_thread=False)
        self.con.execute(
            "CREATE VIRTUAL TABLE p USING fts5(asin UNINDEXED, title, categories, features, "
            "details, store, description, tokenize='unicode61 remove_diacritics 2')"
        )
        rows = []
        self.blob: dict[str, str] = {}
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                a = str(d["parent_asin"])
                fields = (self._f(d.get("title")), self._f(d.get("categories")),
                          self._f(d.get("features")), self._f(d.get("details")),
                          self._f(d.get("store")), self._f(d.get("description")))
                rows.append((a, *fields))
                # Normalised token stream per product, for O(1) phrase-coverage checks.
                self.blob[a] = " " + " ".join(raw_toks(" ".join(fields))) + " "
        self.con.executemany("INSERT INTO p VALUES (?,?,?,?,?,?,?)", rows)
        self.con.commit()
        self.n = len(rows)

    def covers(self, asin: str, phrase_tokens: str) -> bool:
        """True if the product's text contains this token sequence contiguously."""
        return f" {phrase_tokens} " in self.blob.get(asin, "")

    @staticmethod
    def _f(v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, dict):
            return " ".join(f"{k} {x}" for k, x in v.items())
        if isinstance(v, list):
            return " ".join(str(x) for x in v)
        return str(v)

    def search(self, expr: str, limit: int) -> list[str]:
        if not expr or not expr.strip(' "'):
            return []
        try:
            return [r[0] for r in self.con.execute(
                f"SELECT asin FROM p WHERE p MATCH ? ORDER BY {self.W} LIMIT ?", (expr, limit)).fetchall()]
        except sqlite3.OperationalError:
            return []


class Session:
    __slots__ = ("category", "constraints", "raw_tokens", "turn")

    def __init__(self) -> None:
        self.category: str = ""
        self.constraints: list[str] = []
        self.raw_tokens: list[str] = []
        self.turn = 0

    def observe(self, msg: str) -> None:
        """Accumulate. Structured extraction when the template matches, raw tokens always."""
        self.turn += 1
        if not self.category:
            m = PAT_LOOKING.search(msg)
            if m:
                self.category = m.group(1).strip()
        new: list[str] = []
        for pat in (PAT_REQUIREMENT, PAT_MATTERS, PAT_OVERRIDE):
            m = pat.search(msg)
            if m:
                new.extend(part.strip() for part in m.group(1).split(";") if part.strip())
        if not new and not PAT_NOINFO.search(msg):
            # unrecognised phrasing: keep the whole message as evidence rather than dropping it
            residual = PAT_LOOKING.sub("", msg).strip(" .,")
            if len(toks(residual)) >= 2:
                new.append(residual)
        for c in new:
            if c not in self.constraints:
                self.constraints.append(c)
        for t in toks(msg):
            if t not in self.raw_tokens:
                self.raw_tokens.append(t)


# --------------------------------------------------------------------------- variants

class V1_Stateful:
    """Session accumulation only; still bag-of-terms OR, still ask_attribute=None."""
    ASK = None
    PHRASE = False
    LADDER = False

    def __init__(self, index: Index):
        self.ix = index
        self.s: dict[str, Session] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.s[session_id] = Session()

    def _query(self, st: Session, top_k: int) -> list[str]:
        terms = list(dict.fromkeys(self.ix._f(st.category).split() + st.raw_tokens))
        terms = [t.lower() for t in terms if t.lower() not in STOP][:40]
        return self.ix.search(" OR ".join(f'"{t}"' for t in toks(" ".join(terms))), top_k)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        st = self.s.setdefault(session_id, Session())
        st.observe(user_message)
        return {"message": "Here are the closest matches.", "ask_attribute": self.ASK,
                "recommendations": [{"parent_asin": a} for a in self._query(st, top_k)],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0}}


class V2_Probe(V1_Stateful):
    """+ maximal probe: ask_attribute='other' bypasses classify_constraint entirely."""
    ASK = "other"


class V3_Phrase(V2_Probe):
    """+ phrase queries: constraints are verbatim substrings of the target's own text."""

    def _query(self, st: Session, top_k: int) -> list[str]:
        parts = [phrase_of(st.category)] + [phrase_of(c) for c in st.constraints]
        parts = [p for p in parts if p.strip(' "')]
        if not parts:
            return self.ix.search(" OR ".join(f'"{t}"' for t in st.raw_tokens[:40]), top_k)
        return self.ix.search(" OR ".join(parts), top_k)


class V4_Coverage(V3_Phrase):
    """+ constraint-coverage scoring over a wide recall pool.

    Pass 3 showed strict conjunctive AND is extremely selective (median 1 match)
    but returns EMPTY on ~20% of sessions -- tokenisation and 180-char truncation
    drift mean a constraint sometimes has no exact home in the index. So instead of
    committing to one boolean form, retrieve a WIDE pool and rank it by how many
    constraint phrases each product actually contains. Coverage degrades gracefully
    where AND would fall off a cliff.
    """
    POOL = 400

    def _query(self, st: Session, top_k: int) -> list[str]:
        cat = phrase_of(st.category)
        cons = [phrase_of(c) for c in st.constraints]
        cons = [c for c in cons if c.strip(' "')]

        pool: list[str] = []
        seen: set[str] = set()

        def add(expr: str, lim: int) -> None:
            for a in self.ix.search(expr, lim):
                if a not in seen:
                    seen.add(a)
                    pool.append(a)

        # widest-first recall: exact conjunctions, then phrase OR, then token OR
        if cons:
            add(" AND ".join(cons + ([cat] if cat.strip(' "') else [])), self.POOL)
            add(" AND ".join(cons), self.POOL)
            add(" OR ".join(cons + ([cat] if cat.strip(' "') else [])), self.POOL)
        if cat.strip(' "'):
            add(cat, self.POOL)
        if st.raw_tokens:
            add(" OR ".join(f'"{t}"' for t in st.raw_tokens[:40]), self.POOL)
        if not pool:
            return []

        # rank pool by phrase coverage; BM25 order within the pool breaks ties
        want = [c.strip('"') for c in cons]
        catp = cat.strip('"')
        base = {a: i for i, a in enumerate(pool)}

        def score(a: str) -> tuple[int, int, int]:
            cov = sum(1 for w in want if self.ix.covers(a, w))
            catc = 1 if catp and self.ix.covers(a, catp) else 0
            return (-cov, -catc, base[a])

        return sorted(pool, key=score)[:top_k]


def run(name: str, agent, samples, cid, cats, prods) -> dict:
    t0 = time.time()
    r = evaluate(agent, samples, cid, cats, prods)
    r["_secs"] = time.time() - t0
    print(f"{name:<28} HR@10 {r['hit_rate_at_10']:>6.1%}  MRR {r['mrr']:>6.3f}  "
          f"MTTC {r['mttc']:>5.2f}  Eff {r['efficiency']:>5.3f}  "
          f"SCORE {r['recommended_technical_score']:>6.4f}   ({r['_secs']:.0f}s)")
    return r


if __name__ == "__main__":
    print("loading catalog + building shared index ...")
    samples = load_jsonl(PUBLIC)
    cid, cats, prods = catalog_index(CATALOG)
    ix = Index(CATALOG)
    print(f"indexed {ix.n:,} products\n")

    print(f"{'variant':<28} {'':<8}{'':<8}{'':<8}{'':<8}")
    print("-" * 100)
    results = {}
    results["V0 starter BM25"] = run("V0 starter BM25", StarterAgent(CATALOG), samples, cid, cats, prods)
    results["V1 +session state"] = run("V1 +session state", V1_Stateful(ix), samples, cid, cats, prods)
    results["V2 +probe 'other'"] = run("V2 +probe 'other'", V2_Probe(ix), samples, cid, cats, prods)
    results["V3 +phrase queries"] = run("V3 +phrase queries", V3_Phrase(ix), samples, cid, cats, prods)
    results["V4 +coverage rank"] = run("V4 +coverage rank", V4_Coverage(ix), samples, cid, cats, prods)

    print("\nper-scenario breakdown for V4:")
    for scen, m in results["V4 +coverage rank"]["scenario_metrics"].items():
        print(f"  {scen:<18} n={m['sample_count']:<4} HR@10 {m['hit_rate_at_10']:>6.1%}  "
              f"MRR {m['mrr']:>6.3f}  MTTC {m['mttc']:>5.2f}")

    slim = {k: {kk: vv for kk, vv in v.items() if kk not in ("sessions",)} for k, v in results.items()}
    Path(ROOT / "notes" / "eda" / "out_04.json").write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")
    print("\n[saved] notes/eda/out_04.json")
