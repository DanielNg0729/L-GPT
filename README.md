# TikTok TechJam 2026 — Track 4: Shopping Copilot

AI conversational search & recommendation agent trên catalog Amazon (50.000 sản phẩm,
`Clothing_Shoes_and_Jewelry`). Agent phải tìm ra đúng sản phẩm mục tiêu đang bị giấu
trong **tối đa 10 lượt hội thoại**.

## Trạng thái

- [x] Tải + verify toàn bộ tài nguyên BTC cấp → [provided/](provided/)
- [x] Reproduce baseline BM25 bằng evaluator chính thức — **khớp 100%** số BTC công bố
- [ ] Intent router (Buying / Browsing)
- [ ] Multi-route retrieval (BM25 + category + vector, in-memory)
- [ ] Dialog state machine (slot accumulation + intent override)
- [ ] LLM semantic ranking
- [ ] Tối ưu MTTC

## Baseline đã chạy được

`starter/agent.py` (BM25 yếu) trên 200 public session:

| Metric | Số của mình | BTC công bố |
|---|---|---|
| Hit Rate@10 | 0.125 | 0.125 |
| MRR | 0.068034 | 0.068034 |
| MTTC | 9.81 | 9.81 |
| Efficiency | 0.119 | 0.119 |
| **TechnicalScore** | **0.10671** | **0.10671** |

Tách theo scenario — chỗ này lộ ra điểm yếu chính của baseline:

| Scenario | n | Hit@10 | MRR | MTTC |
|---|---|---|---|---|
| buying | 80 | 0.2375 | 0.1265 | 8.63 |
| intent_override | 30 | 0.1333 | 0.1042 | 10.07 |
| browsing | 80 | **0.025** | 0.0045 | 10.75 |
| boundary | 10 | **0.000** | 0.0000 | 11.00 |

> **browsing + boundary = 90/200 session gần như trắng điểm.** Đây là chỗ ăn điểm lớn nhất:
> chỉ cần kéo browsing từ 0.025 lên ngang buying là TechnicalScore gần gấp đôi.

## Chạy lại

```bash
cd provided && ./fetch.sh                          # tải catalog (không nằm trong git)
cd techjam-conversational-search
python3 -m evaluator.local_evaluator               # -> results.json
```

Không cần cài dependency — evaluator và starter agent chỉ dùng stdlib.

## Tài liệu

- [TRACK4_SHOPPING_COPILOT.md](TRACK4_SHOPPING_COPILOT.md) — tóm tắt đề bài, rule, tiêu chí chấm
- [TechJam2026_Tom_tat.md](TechJam2026_Tom_tat.md) — tóm tắt chung các track
- [provided/README.md](provided/README.md) — nguồn gốc & checksum tài nguyên BTC

## Rule dễ mất điểm

- Vượt **10 turn** → session đó **0 điểm**. Phải có hard counter trong agent loop.
- Catalog **read-only** — cấm mutate, cấm inject ASIN giả.
- Vector index phải **in-memory**, không dựng service ngoài.
- **Không commit API key.** BTC không cấp và không hoàn tiền API credit.
