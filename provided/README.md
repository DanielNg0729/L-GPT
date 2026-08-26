# `provided/` — Tài nguyên do BTC TikTok TechJam 2026 cấp

> **READ-ONLY.** Không sửa bất cứ file nào trong thư mục này. Code của team viết ở ngoài,
> import vào. Catalog là read-only theo rule (cấm mutate cấu trúc, cấm inject ASIN giả).

## Nguồn

| | |
|---|---|
| Repo BTC | https://github.com/TechJam2026/techjam-conversational-search |
| Release | `participant-kit` (published 2026-08-24) |
| Upstream commit | xem `UPSTREAM_COMMIT.txt` |
| Data gốc | Amazon Reviews 2023 — https://amazon-reviews-2023.github.io/ |

## Cấu trúc

```
provided/
├── SHA256SUMS                       # checksum chính chủ BTC
├── UPSTREAM_COMMIT.txt              # commit repo BTC đã snapshot
├── fetch.sh                         # tải lại + verify (dùng khi clone mới)
├── release/                         # asset nguyên bản (KHÔNG commit — xem .gitignore)
│   ├── catalog.jsonl.gz             # 18MB
│   └── techjam-participant-kit.zip  # 18MB
└── techjam-conversational-search/   # kit đã giải nén (= repo BTC + catalog)
    ├── README.md                    # bản trong repo, mới hơn bản trong zip
    ├── DATA_ATTRIBUTION.md
    ├── data/
    │   ├── catalog.jsonl            # 50.000 sản phẩm, 60MB — KHÔNG commit
    │   └── public_set.jsonl         # 200 dev session có nhãn
    ├── docs/                        # spec, API contract, eval config, baseline, rules
    ├── evaluator/local_evaluator.py # evaluator chính thức, deterministic
    ├── starter/agent.py             # BM25 baseline yếu
    └── tests/test_evaluator.py      # chỉ có trong repo, không có trong zip
```

## Sau khi clone repo này

`catalog.jsonl`, `catalog.jsonl.gz`, `techjam-participant-kit.zip` bị gitignore (tổng ~96MB).
Chạy để lấy lại:

```bash
cd provided && ./fetch.sh
```

Cần `gh` CLI đã đăng nhập. Script tự verify SHA-256 và kiểm tra đúng 50.000 dòng.

## Đã verify

- `shasum -a 256 -c SHA256SUMS` → cả 2 asset **OK**
- `catalog.jsonl` trong zip **trùng hash** với `catalog.jsonl.gz` giải nén
- `catalog.jsonl` = 50.000 dòng · `public_set.jsonl` = 200 dòng
- Phân bố scenario: buying 80 · browsing 80 · intent_override 30 · boundary 10

## Lưu ý quan trọng

**README trong zip khác README trong repo** — bản repo mới hơn và đây là bản đang giữ:

> ~~The organizer may reimburse model costs through prizes instead of issuing API keys.~~
> → **The organizer does not provide or reimburse model API credits; teams are responsible
> for any costs incurred through optional external services.**

Tức là **BTC không cấp và không hoàn tiền API credit**. Không bắt buộc dùng LLM trả phí.

**Không bao giờ** đặt API key, private eval data, hay output của agent vào thư mục này.
