#!/usr/bin/env python3
"""Data exploration cho catalog TechJam 2026 Track 4.

Chạy:  python3 scripts/explore_catalog.py
Ra:    docs/DATA_PROFILE.md  +  docs/data_profile.json  (+ report ra stdout)

Chỉ dùng stdlib. Một pass qua 50k dòng cho phần schema, sample cho phần
phân tích nặng (intent_card của evaluator).
"""
from __future__ import annotations

import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "provided" / "techjam-conversational-search"
CATALOG = KIT / "data" / "catalog.jsonl"
PUBLIC = KIT / "data" / "public_set.jsonl"
OUT_MD = ROOT / "docs" / "DATA_PROFILE.md"
OUT_JSON = ROOT / "docs" / "data_profile.json"

# dùng lại chính logic của evaluator để không bị lệch giả định
sys.path.insert(0, str(KIT))
from evaluator.local_evaluator import (  # noqa: E402
    intent_card, classify_constraint, searchable_text, coarse_category,
    MATERIAL_RE, COLOR_RE,
)

SAMPLE_N = 4000
SEED = 20260828


def pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 2) if total else 0.0


def describe(values: list[float]) -> dict:
    if not values:
        return {}
    s = sorted(values)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    return {
        "n": len(s), "min": round(s[0], 2), "p10": round(q(0.10), 2),
        "p25": round(q(0.25), 2), "median": round(statistics.median(s), 2),
        "p75": round(q(0.75), 2), "p90": round(q(0.90), 2),
        "p99": round(q(0.99), 2), "max": round(s[-1], 2),
        "mean": round(statistics.fmean(s), 2),
    }


def main() -> None:
    rng = random.Random(SEED)
    prof: dict = {}

    top_present = Counter()
    top_null = Counter()
    top_types = defaultdict(Counter)
    detail_keys = Counter()
    detail_key_count: list[float] = []
    detail_value_card = defaultdict(set)     # key -> set giá trị (cắt bớt)
    cat_depth: list[float] = []
    cat_l2 = Counter()
    cat_leaf = Counter()
    store_c = Counter()
    prices: list[float] = []
    price_null = 0
    ratings: list[float] = []
    rating_n: list[float] = []
    title_len: list[float] = []
    feat_n: list[float] = []
    desc_len: list[float] = []
    has_material = 0
    has_color = 0
    has_size_hint = 0
    total = 0
    sample: list[dict] = []
    products: dict[str, dict] = {}
    targets_needed: set[str] = set()

    if PUBLIC.exists():
        for line in PUBLIC.read_text().splitlines():
            if line.strip():
                targets_needed.add(json.loads(line)["ground_truth"]["parent_asin"])

    size_re = re.compile(r"\b(size|sizing|width|wide|narrow|small|medium|large|xl|xxl)\b", re.I)

    with CATALOG.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            p = json.loads(line)
            total += 1

            for k, v in p.items():
                top_present[k] += 1
                if v in (None, "", [], {}):
                    top_null[k] += 1
                top_types[k][type(v).__name__] += 1

            d = p.get("details") or {}
            if isinstance(d, dict):
                detail_key_count.append(len(d))
                for k, v in d.items():
                    detail_keys[k] += 1
                    if len(detail_value_card[k]) < 5000:
                        detail_value_card[k].add(str(v)[:80])

            cats = p.get("categories") or []
            if cats:
                cat_depth.append(len(cats))
                if len(cats) >= 2:
                    cat_l2[cats[1]] += 1
                cat_leaf[" > ".join(cats[-2:])] += 1

            if p.get("store"):
                store_c[p["store"]] += 1

            price = p.get("price")
            if isinstance(price, (int, float)):
                prices.append(float(price))
            else:
                price_null += 1

            if isinstance(p.get("average_rating"), (int, float)):
                ratings.append(float(p["average_rating"]))
            if isinstance(p.get("rating_number"), int):
                rating_n.append(float(p["rating_number"]))

            title_len.append(len(str(p.get("title") or "").split()))
            feats = p.get("features") or []
            feat_n.append(len(feats) if isinstance(feats, list) else 0)
            desc = p.get("description") or []
            desc_len.append(len(" ".join(desc).split()) if isinstance(desc, list) else 0)

            txt = searchable_text(p)
            if MATERIAL_RE.search(txt):
                has_material += 1
            if COLOR_RE.search(txt):
                has_color += 1
            if size_re.search(txt):
                has_size_hint += 1

            # reservoir sample + giữ lại product là target của public set
            if len(sample) < SAMPLE_N:
                sample.append(p)
            elif rng.random() < SAMPLE_N / total:
                sample[rng.randrange(SAMPLE_N)] = p
            if p.get("parent_asin") in targets_needed:
                products[p["parent_asin"]] = p

    # ---------- schema ----------
    prof["catalog_size"] = total
    prof["top_level_fields"] = {
        k: {
            "present_pct": pct(v, total),
            "empty_pct": pct(top_null[k], total),
            "types": dict(top_types[k]),
        }
        for k, v in top_present.most_common()
    }

    # ---------- details = không gian attribute thật ----------
    prof["details"] = {
        "distinct_keys": len(detail_keys),
        "keys_per_product": describe(detail_key_count),
        "top_keys": [
            {
                "key": k, "coverage_pct": pct(c, total),
                "distinct_values_sampled": len(detail_value_card[k]),
            }
            for k, c in detail_keys.most_common(40)
        ],
        "keys_with_coverage_over_10pct": sum(1 for c in detail_keys.values() if c > total * 0.10),
        "keys_with_coverage_over_1pct": sum(1 for c in detail_keys.values() if c > total * 0.01),
        "long_tail_under_100_items": sum(1 for c in detail_keys.values() if c < 100),
    }

    # ---------- categories ----------
    prof["categories"] = {
        "depth": describe(cat_depth),
        "distinct_leaf_pairs": len(cat_leaf),
        "top_level2": cat_l2.most_common(15),
        "top_leaf": cat_leaf.most_common(20),
        "coarse_category_distinct": None,
    }

    # ---------- numeric ----------
    prof["price"] = {"null_pct": pct(price_null, total), "dist": describe(prices)}
    prof["average_rating"] = describe(ratings)
    prof["rating_number"] = describe(rating_n)
    prof["text"] = {
        "title_words": describe(title_len),
        "features_count": describe(feat_n),
        "description_words": describe(desc_len),
    }
    prof["store"] = {
        "distinct": len(store_c),
        "top": store_c.most_common(15),
        "singleton_pct": pct(sum(1 for c in store_c.values() if c == 1), len(store_c)),
    }
    prof["signal_coverage"] = {
        "material_regex_pct": pct(has_material, total),
        "color_regex_pct": pct(has_color, total),
        "size_hint_pct": pct(has_size_hint, total),
    }

    # ---------- góc nhìn evaluator: user sẽ tiết lộ cái gì ----------
    def card_buckets(items: list[dict]) -> dict:
        bucket = Counter()
        hard_n: list[float] = []
        soft_n: list[float] = []
        per_card_buckets = Counter()
        for p in items:
            card = intent_card(p)
            cons = list(card["hard_constraints"]) + list(card["soft_preferences"])
            hard_n.append(len(card["hard_constraints"]))
            soft_n.append(len(card["soft_preferences"]))
            bs = [classify_constraint(str(c)) for c in cons]
            bucket.update(bs)
            per_card_buckets[len(set(bs))] += 1
        tot = sum(bucket.values())
        return {
            "constraint_bucket_share_pct": {
                k: pct(v, tot) for k, v in bucket.most_common()
            },
            "hard_constraints": describe(hard_n),
            "soft_preferences": describe(soft_n),
            "distinct_buckets_per_card": dict(sorted(per_card_buckets.items())),
        }

    prof["intent_card_sampled"] = card_buckets(sample)
    if products:
        prof["intent_card_public_targets"] = card_buckets(list(products.values()))
        prof["public_targets_found"] = f"{len(products)}/{len(targets_needed)}"
        prof["public_target_coarse_category"] = Counter(
            coarse_category(p.get("categories") or []) for p in products.values()
        ).most_common(15)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(prof, ensure_ascii=False, indent=2))

    # ---------- report ----------
    L: list[str] = []
    A = L.append
    A("# Data profile — catalog Track 4\n")
    A(f"`{total:,}` sản phẩm · sinh bởi `scripts/explore_catalog.py`\n")

    A("## 1. Top-level fields — chỉ có 11\n")
    A("| field | present % | empty/null % | types |")
    A("|---|---|---|---|")
    for k, v in prof["top_level_fields"].items():
        A(f"| `{k}` | {v['present_pct']} | {v['empty_pct']} | {', '.join(v['types'])} |")

    d = prof["details"]
    A(f"\n## 2. `details` — không gian attribute thật: **{d['distinct_keys']} key phân biệt**\n")
    A(f"- Số key / sản phẩm: median **{d['keys_per_product']['median']}**, "
      f"p90 **{d['keys_per_product']['p90']}**, max **{d['keys_per_product']['max']}**")
    A(f"- Key phủ >10% catalog: **{d['keys_with_coverage_over_10pct']}** · "
      f">1%: **{d['keys_with_coverage_over_1pct']}** · "
      f"đuôi dài <100 item: **{d['long_tail_under_100_items']}**\n")
    A("| details key | coverage % | distinct values |")
    A("|---|---|---|")
    for row in d["top_keys"]:
        A(f"| `{row['key']}` | {row['coverage_pct']} | {row['distinct_values_sampled']} |")

    c = prof["categories"]
    A(f"\n## 3. Categories\n")
    A(f"- Độ sâu: median **{c['depth']['median']}**, max **{c['depth']['max']}**")
    A(f"- Số cặp lá phân biệt: **{c['distinct_leaf_pairs']}**\n")
    A("| level-2 | n |")
    A("|---|---|")
    for k, v in c["top_level2"]:
        A(f"| {k} | {v:,} |")

    A(f"\n## 4. Số liệu\n")
    A(f"- **price null: {prof['price']['null_pct']}%** → budget constraint chỉ tồn tại ở phần còn lại")
    A(f"- price: {prof['price']['dist']}")
    A(f"- average_rating: {prof['average_rating']}")
    A(f"- rating_number: {prof['rating_number']}")
    A(f"- title words: {prof['text']['title_words']}")
    A(f"- features count: {prof['text']['features_count']}")
    A(f"- description words: {prof['text']['description_words']}")
    A(f"- store phân biệt: **{prof['store']['distinct']:,}** "
      f"({prof['store']['singleton_pct']}% chỉ có 1 sản phẩm)")
    A(f"- regex coverage: material {prof['signal_coverage']['material_regex_pct']}% · "
      f"color {prof['signal_coverage']['color_regex_pct']}% · "
      f"size-hint {prof['signal_coverage']['size_hint_pct']}%")

    A("\n## 5. Góc nhìn evaluator — user thực sự tiết lộ cái gì\n")
    for label, key in (("sample ngẫu nhiên", "intent_card_sampled"),
                       ("200 target của public set", "intent_card_public_targets")):
        if key not in prof:
            continue
        b = prof[key]
        A(f"**{label}** — phân bố bucket của constraint (`classify_constraint`):\n")
        A("| bucket | % constraint |")
        A("|---|---|")
        for k, v in b["constraint_bucket_share_pct"].items():
            A(f"| `{k}` | {v} |")
        A(f"\nSố bucket phân biệt / intent card: `{b['distinct_buckets_per_card']}`\n")

    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n-> {OUT_MD}\n-> {OUT_JSON}")


if __name__ == "__main__":
    main()
