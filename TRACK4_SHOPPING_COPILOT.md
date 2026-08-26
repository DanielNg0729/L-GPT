# Track 4 — Shopping Copilot: AI Conversational Search and Recommendations

> TikTok TechJam 2026 · Technical Workshop Webinar: **28/08/2026, 4:00–4:45pm SGT**
> Tóm tắt từ bản Early Bird Access — Tracks & Problem Statements.

---

## 1. Bối cảnh & bài toán

Search e-commerce truyền thống dựa nặng vào **keyword matching tĩnh**, nên không bắt được:

- Dòng chảy tâm lý người mua thay đổi liên tục trong hội thoại
- Sự khác biệt giữa **browsing** (lướt, chưa rõ nhu cầu) và **buying** (ý định mua cao, ràng buộc cứng)

Nhiệm vụ: xây một **shopping agent hội thoại** trên catalog Amazon, hiểu ngữ cảnh sâu, tự điều chỉnh
workflow lúc runtime, và hiệu quả về mặt thương mại — đo bằng chính record mua hàng thật trong dataset.

---

## 2. Bốn trụ cột bắt buộc

### I. Core Architecture — Intent Routing & Hybrid Pipeline

**Dual-Track Routing** — phát hiện intent ngay từ đầu rồi rẽ nhánh:

| Track | Khi nào | Cách xử lý |
|---|---|---|
| **Buying** | Ý định mua rõ, có ràng buộc cứng (size, màu, giá, brand) | Filter track chính xác cao, khoá hard constraint |
| **Browsing** | Mở, mơ hồ, khám phá | Dense retrieval đa dạng, mở khoá cross-category scenario matching |

**Pipeline Base** — luồng dữ liệu **in-memory**:

```
Multi-Route Retrieval  →  LLM Semantic Ranking
  ├── keyword (BM25)
  ├── category
  └── vector similarity
```

### II. Dialog Strategy — Multi-Turn Scenario Evolution

**Dynamic State Machine** — conversational state tracker xử lý được 2 tình huống ngược nhau:

- **Information Accumulation** — slot tăng dần qua từng lượt (thêm ràng buộc)
- **Intent Override** — người dùng đổi ý đột ngột → **xoá và ghi lại slot**, không được cộng dồn sai

**Proactive Guidance** — khi gặp **Over-Generality** (candidate pool quá lớn):
cắt retrieval ngay lập tức và **chủ động sinh câu hỏi làm rõ có cấu trúc** để hội tụ nhanh.

### III. Self-Evolution — Dynamic Context Programming

- **Runtime Adaptation** — *Personalized Context Distillation* từ lịch sử hội thoại,
  cập nhật liên tục **short-term session state** + **long-term user profile**
- **Adaptive Orchestration** — dùng dynamic Context Programming để **re-orchestrate workflow lúc runtime**
  và align strategy, để agent tự tinh chỉnh logic guidance của chính nó qua từng vòng

### IV. Evaluation Matrix — Product & Efficiency

Neo vào **record sản phẩm thực sự được mua** trong dataset Amazon:

| Chiều | Metric | Đo cái gì |
|---|---|---|
| **Coverage** | Hit Rate@K | Khả năng recall & bao phủ catalog ở giai đoạn retrieval |
| **Precision** | MRR / Top-K Hit Rate | LLM đẩy đúng món đã mua lên **top đầu** danh sách |
| **Efficiency** | **MTTC** (Mean Turns to Conversion) | Thưởng nặng hệ thống dẫn user tới đúng sản phẩm trong **ít lượt hơn**; phạt tải nhận thức thừa |

> MTTC là điểm khác biệt lớn nhất so với một bài retrieval thông thường:
> hỏi thêm 1 câu clarification phải **đáng giá** — nếu không, nó chỉ làm tụt điểm.

---

## 3. Constraints & Scope

### ✅ In scope
- Module intent-detection nhạy, chia traffic thành "Buying" / "Browsing"
- Heterogeneous retrieval routing: weights, custom dynamic truncation, **slot decay theo thời gian**
- Runtime-adaptive memory layer cho personalized context distillation
- Fine-tune prompt strategy / local scoring logic ở stage LLM ranking để nén decision path

### ❌ Out of scope
- **UI/UX** — chấm hoàn toàn qua automated backend API + headless pipeline
- Train hoặc full-parameter fine-tune base foundational LLM
- Deploy vector DB cluster công nghiệp nặng bên ngoài — **phải chạy hoàn toàn in-memory**
- Multi-modal — chỉ text catalog, structured metadata, text dialog

### ⚠️ Limits (dễ mất điểm nhất)
- **Max 10 turns/session — vượt là bị forced termination và ZERO SCORE**
- **Catalog read-only** — cấm mutate cấu trúc, cấm inject ASIN giả

### Allowed assumptions
- Input đã là text sạch (không cần lo typo, spelling correction, ASR noise)
- Catalog, giá, cây category **tĩnh** suốt hackathon
- Mỗi session là 1 user độc lập (không cần lo concurrency nhiều user)

---

## 4. Tài nguyên & dữ liệu

Bộ kit đóng băng, reproducible, dẫn xuất từ **Amazon Reviews 2023**.

**Competition data**
- Catalog đóng băng **50.000 sản phẩm**, category `Clothing_Shoes_and_Jewelry`
- **200 labeled public dev sessions** — test & iterate local
- **800 private sessions** giữ bởi BTC để chấm cuối
- Public và private dùng **user và target product tách biệt hoàn toàn**

**Participant resources**
- Starter agent **BM25 yếu** (Python)
- **Local evaluator deterministic**: Hit Rate@10, MRR, MTTC, Efficiency, và `TechnicalScore` tổng hợp
- Python Agent interface + machine-readable API contract
- Evaluation config, baseline results reproducible, data docs, submission rules
- File SHA256 checksum để verify catalog tải về

**Links**
- Repo: https://github.com/TechJam2026/techjam-conversational-search
- Participant kit: https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit
- Data gốc: https://amazon-reviews-2023.github.io/

Được phép thay/sửa starter agent nhưng vẫn phải dùng official local evaluator.
Kit hỗ trợ: keyword retrieval, rule-based, dense retrieval, hybrid retrieval, reranking, local model, external model API.

> ⚠️ **BTC KHÔNG cấp** hosted model access, API key, model token hay third-party credit.
> **Không bắt buộc dùng LLM trả phí.** Dùng service ngoài thì tự lo credential + chi phí,
> và **tuyệt đối không commit secret vào repo**.
> Không cần tải/dựng lại toàn bộ dataset Amazon Reviews 2023 gốc.

---

## 5. Deliverables

1. **Written Project Description (Devpost)** — giải pháp giải quyết problem statement thế nào; development tools (VSCode, Colab, Jupyter…); APIs dùng; libraries & frameworks; datasets & assets.
2. **Public GitHub repository** — code có cấu trúc rõ, comment đầy đủ; README gồm: project overview, setup & installation, các bước reproduce kết quả, reflection về limitation + điều sẽ cải thiện nếu có thêm thời gian, đóng góp của từng thành viên.
3. **Demo video** — chạy end-to-end, upload YouTube **public**, link trong Devpost, không dùng trademark/nội dung có bản quyền của bên thứ ba.
   *Track backend/NLP không có front-end thì được nộp walkthrough API usage / inference examples / result analysis.*

---

## 6. Judging Criteria

| Tiêu chí | Trọng số | Nội dung |
|---|---|---|
| **Technical Execution** | 35% | Engineering fundamentals vững, code có cấu trúc, architecture có suy nghĩ, dùng API/model hiệu quả, demo chạy ổn định, độ phức tạp kỹ thuật phản ánh quyết định có chủ đích |
| **Innovation & Problem Insight** | 20% | Độc đáo trong cả ý tưởng lẫn cách tiếp cận; sắc sảo trong việc framing vấn đề — vì sao nó quan trọng và giải pháp giải quyết trực diện ra sao |
| **Impact & Relevance** | 20% | Tiềm năng tạo giá trị thật cho user/stakeholder, có reach và lợi ích cụ thể, vượt ra ngoài phạm vi đề hackathon |
| **Feasibility & Practicality** | 15% | Thực tế, build được vượt mức prototype; resource dùng hợp lý, architecture trụ được điều kiện thực, implementation có cơ sở chứ không nói suông |
| **Presentation & Communication** | 10% | *[Final Event only]* Pitch mạch lạc từ problem → solution → potential, trả lời câu hỏi có chiều sâu |

---

## 7. Checklist thực thi

**Ngày 1 — Baseline & khung**
- [ ] Clone participant kit, verify catalog bằng SHA256
- [ ] Chạy được starter BM25 agent + local evaluator, ghi lại điểm baseline (Hit Rate@10 / MRR / MTTC)
- [ ] Dựng harness chạy 200 public session và ra bảng số trong 1 lệnh
- [ ] Chốt kiến trúc: intent router → multi-route retrieval → LLM rank

**Ngày 2 — Bốn trụ cột**
- [ ] Intent classifier Buying vs Browsing (đo riêng accuracy của nó)
- [ ] Multi-route retrieval: BM25 + category filter + vector (in-memory, ví dụ FAISS/numpy)
- [ ] Dialog state machine: slot accumulation **và** intent override (viết test riêng cho case override)
- [ ] Ngưỡng over-generality → sinh clarification question có cấu trúc
- [ ] LLM semantic ranking + prompt tuning

**Ngày 3 — Tối ưu điểm & nộp**
- [ ] **Tối ưu MTTC**: ít lượt nhất có thể; ablation "hỏi thêm 1 câu có đáng không"
- [ ] Context distillation / user profile ngắn hạn + dài hạn
- [ ] Kiểm tra **không session nào vượt 10 turn** (guard cứng trong code)
- [ ] Quét repo đảm bảo **không lộ API key**
- [ ] README + demo video + Devpost description

---

## 8. Rủi ro cần canh

| Rủi ro | Hệ quả | Cách phòng |
|---|---|---|
| Session vượt 10 turn | **0 điểm session đó** | Hard counter trong agent loop, buộc trả kết quả ở turn 9 |
| Hỏi clarification quá nhiều | MTTC tệ → tụt Efficiency | Chỉ hỏi khi pool thực sự quá rộng; đo delta trước/sau |
| Overfit 200 public session | Rớt trên 800 private session | Cross-validate, tránh hard-code rule theo sample cụ thể |
| Phụ thuộc LLM API trả phí | Hết credit giữa chừng / không reproduce được | Có fallback local scoring; cache LLM response |
| Commit `.env` / API key | Vi phạm rule, mất điểm | `.gitignore` + quét repo trước khi push |
| Vector index nặng | Vi phạm "in-memory only" | Embedding nhẹ, index in-process, không dựng service ngoài |
