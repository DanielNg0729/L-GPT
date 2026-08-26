# TikTok TechJam 2026 — Tổng hợp 5 Tracks & Problem Statements

> Nguồn: `TikTok_TechJam_2026_Tracks_and_Problem_Statements.pdf` (Early Bird Access, Feishu Docs).
> Bản PDF có **đầy đủ** nội dung cả 5 track (bản HTML chỉ có Track 1 do Feishu lazy-load).

## Webinar (bắt buộc nên xem trước khi chọn track)

- **Ngày:** Thứ Sáu 28/08/2026, 1–6pm SGT (GMT+8), mỗi track 45 phút
- **Link:** https://vc-my.larkoffice.com/j/484622806 (link chính thức share sau public release)

| Giờ | Track |
|---|---|
| 1:00–1:45pm | 1. Agent Launchpad: Design and Build Lightweight Agent Middleware |
| 2:00–2:45pm | 2. Autonomous ML Research Agent for Recommender Systems |
| 3:00–3:45pm | 3. Implement a GPU Kernel for a Transformer Layer |
| 4:00–4:45pm | 4. Shopping Copilot: AI Conversational Search and Recommendations |
| 5:00–5:45pm | 5. Robust Detection of AI-Generated Images Under Real-World Transformations |

## So sánh nhanh

| # | Track | Kỹ năng chính | Chấm điểm chính | Độ khó / rủi ro |
|---|---|---|---|---|
| 1 | Agent Middleware | Full-stack TS (React + Fastify), Docker, security/observability | 40% end-to-end middleware behavior | Vừa. Có starter kit chạy sẵn, đề mở → dễ vào nhưng phải chọn scope khôn |
| 2 | ML Research Agent | LLM agent + RecSys (CTR/CVR), MLOps | Delta điểm trên hidden test (đo tự động) | **Cao nhất.** Cần GPU, data ~80M samples, thời gian train |
| 3 | GPU Kernel | CUDA/Triton, profiling, tối ưu | 35% Technical Execution | Cao về chuyên môn, nhưng scope gọn và rõ; cần GPU riêng |
| 4 | Shopping Copilot | RAG / hybrid retrieval, dialog state, prompt eng | 35% Technical Execution + evaluator tự động | **Vừa nhất.** In-memory, không cần GPU, có evaluator local |
| 5 | AIGC Image Detection | CV, robustness, augmentation | 35% Technical Execution | Vừa. Dataset public sẵn, giới hạn model < 2B params |

Tiêu chí chấm chung cho track 3, 4, 5 (và gần giống track 1, 2):
Technical Execution 35% · Innovation & Problem Insight 20% · Impact & Relevance 20% ·
Feasibility & Practicality 15% · Presentation & Communication 10% *(chỉ ở Final Event)*.

Deliverables chung cho track 2–5: **Devpost description + public GitHub repo (README đầy đủ) + demo video YouTube public**.

---

# Track 1 — Agent Launchpad: Lightweight Agent Middleware

**Starter kit:** https://github.com/RrankPyramid/CodeJam
**Slogan:** *"Build the missing middleware, not the platform."*

## Bối cảnh
Agent platform cần nhiều hơn một khung chat: operator phải **hiểu chuyện gì đã xảy ra**,
**kiểm soát Agent được truy cập gì**, và **cô lập phần thực thi không an toàn**. Starter kit
đã lo hết phần platform để đội dành trọn 3 ngày cho một bài toán hạ tầng.

## Starter Kit đã có gì
- CRUD/start/stop/delete Agent từ browser, Playground multi-turn, poll Run status bất đồng bộ
- Codex CLI ghi file & chạy lệnh trong workspace riêng của từng Agent, resume session
- Lưu Agent/message/Run metadata vào JSON store local
- Mỗi turn chạy trong container Docker/Colima/Podman dùng-rồi-bỏ
- Kết nối BytePlus ModelArk (Responses-compatible endpoint)
- Deploy tuỳ chọn lên BytePlus ECS (Terraform)

**Cố tình thiếu (phần mình làm):** user identity, trace timeline, audit model, sandbox policy cứng.
Đây là POC single-user; bearer token không phải hệ thống identity; JSON store 1 process;
container thường không phải isolation boundary multi-tenant.

## Phân chia trách nhiệm

| Mảng | Starter Kit | Sinh viên làm |
|---|---|---|
| Product experience | React UI, Agent list, form, lifecycle, Playground, Run status | Giữ baseline chạy; chỉ thêm UI vừa đủ để lộ middleware |
| Control plane | Fastify API, validation, async Runs, AgentService, JSON persistence | Cắm hành vi middleware thật vào backend path |
| Agent Runtime | Codex CLI, session bền, workspace/Agent, container dùng-rồi-bỏ | Cắm middleware vào đúng execution boundary |
| Infrastructure | Docker, Colima, Podman, Compose, ECS, Terraform | Dùng runtime path nhỏ nhất đủ chứng minh. Cloud **optional** |
| Middleware | (trống) | Chọn / chỉnh / kết hợp / **tự sáng tạo** một bộ capability mạch lạc |

## Chạy baseline
Yêu cầu: macOS/Linux · Node ≥ 22, npm ≥ 10 · một container engine · **Ark model API key** + endpoint `ep-...`

```bash
git clone https://github.com/RrankPyramid/CodeJam.git && cd CodeJam
ARK_API_KEY=your-ark-api-key ARK_MODEL=ep-your-endpoint-id npm run poc
# rootless podman: thêm CONTAINER_ENGINE=podman
# mở http://localhost:3000
```
⚠️ `ARK_API_KEY` phải là **Ark model API key**, KHÔNG phải AK/SK tài khoản BytePlus → sai sẽ ra 401.

**Acceptance test baseline:** tạo Agent → gửi task Playground ("Create a TypeScript hello-world CLI,
add a test, run it, and summarize the files you created") → Run xong có phản hồi → follow-up
tiếp tục cùng session → stop/restart Agent workspace vẫn còn.
**Chưa pass baseline thì chưa được bắt tay làm middleware.**
Trước khi nộp bắt buộc chạy `npm run check` (TS check + server tests + production build).

## Yêu cầu thiết kế middleware
- **Giữ nguyên baseline** (CRUD, lifecycle, Playground, persistence, model execution vẫn chạy)
- **Hành vi thật**: phải chạy ở backend / Runtime / data / infra path. Màn hình tĩnh + message hard-code = **không tính**
- **Định nghĩa boundary**: component nào sở hữu quyết định/event, data nào đi qua, fail thì sao
- **Bằng chứng thuyết phục**: demo cả case bình thường **và** case failure/denial/recovery/degraded/abuse
- **Có automated test** cho hành vi middleware lõi (không chỉ test render UI)
- **Không lộ secret** ở source, git history, log, trace, screenshot, browser storage, demo output
- Ưu tiên hạ tầng nhỏ nhất; **local là judging path mặc định, ECS không cộng điểm**

## Các hướng middleware gợi ý (không bắt buộc)
1. **Identity & Authorization** — tách human principal vs Agent principal; per-Agent identity rotate/revoke được; delegated authority scoped + time-bound; policy enforcement ở backend (không phải UI); approval boundary cho hành động rủi ro; action attribution; secret handling & revocation. *Mock identity là chấp nhận được* (VD: chứng minh Agent của User A không đọc được resource của User B). Login screen mà không có server-side authz thì không tính.
2. **Trace, Audit & Observability** — Run = chuỗi span có liên kết, không phải log rời. ID ổn định (agent/version/run/session/trace/span/actor), thời gian + status + error + retry, span category (orchestration, model call, tool call, memory, sandbox, policy decision, approval, cloud op), input/output đã redact, token/cost. UI: Run list + trace detail dạng tree/timeline, filter, tìm bước fail.
3. **Layered Agent Architecture** — Experience / Control Plane / Identity & Policy / Agent Runtime / Execution & Data / Observability / Cloud Resource. Ghi rõ contract giữa các layer.
4. **Threat Modeling & Safety** — credential theft, privilege escalation / confused deputy, prompt injection & tool misuse, sandbox escape, cross-user data exfiltration, runaway execution/cost (timeout, quota, budget, stop control), sensitive trace capture. *Lưu ý: CPU/memory/PID limit sẵn có KHÔNG được tính là capability mới.*
5. **Multi-Agent Coordination** — shared session/topic, turn-selection rule, shared state, event history, timeout/retry. Demo mẫu: nhiều Agent đếm ngược 10→1 trong một hội thoại chung, không trùng không thiếu số.
6. **Tự nghĩ** — lifecycle reconciliation & recovery, memory governance, human-in-the-loop, cost/budget control, provider abstraction, versioning & rollback, tool/model routing, credential exchange, auto-diagnosis & remediation.

## Kế hoạch 3 ngày (đề xuất trong đề)
| Ngày | Mục tiêu | Bằng chứng cuối ngày |
|---|---|---|
| 1 | Chạy & hiểu baseline, chốt bài toán + middleware story, định nghĩa contract, xong backend path đầu tiên | Baseline pass; trigger được 1 hành vi middleware thật từ API hoặc test |
| 2 | Hoàn thiện core path, persist evidence, UI tối thiểu, các case success/failure quan trọng | Kịch bản đầy đủ chạy end-to-end từ browser xuống backend/Runtime/data/infra |
| 3 | Automated tests, xử lý lỗi & cleanup, architecture diagram + README, tập demo | `npm run check` pass; demo gọn trong **3 phút** |

## Deliverables & chấm điểm
**Nộp:** demo live 3 phút · architecture diagram 1 trang (middleware, data flow, trust boundary, điểm enforcement/instrument/recovery) · code repo (setup, rationale, design summary, tests, demo steps, limitations, không secret).

| Tiêu chí | Trọng số |
|---|---|
| End-to-end middleware behavior | **40%** |
| Technical design & integration | 25% |
| Verification & robustness | 20% |
| Demo & reproducibility | 15% |

**Bắt đầu đọc code ở:** `apps/server/src/types.ts`, `app.ts`, `agent-service.ts`, 2 bản `AgentRunner`, rồi `apps/web/src/App.tsx`.

---

# Track 2 — Autonomous ML Research Agent for Recommender Systems

## Đề bài
Xây một **Autonomous ML Research Agent** tự chạy trọn vòng lặp MLE:
đọc đề → EDA → feature engineering → train + tune → evaluate → reflect & revise → lặp lại.
Code của từng stage là do agent tự viết, không được cho sẵn.
Tham chiếu prior work: MLE-Bench (OpenAI), AIDE (Weco AI), AI-Scientist-v2 (Sakana AI).

Agent phải: (1) reproduce official baseline, (2) tự iterate cải tiến pipeline, (3) vượt baseline
trên hidden test. Chỉ được dùng train + validation, **không bao giờ thấy hidden test**.

## Benchmark
| Dataset | Mô tả | Metric | Scale |
|---|---|---|---|
| **AliCCP** (bắt buộc, 100% primary score) | Taobao, funnel impression→click→conversion | CTR AUC (toàn bộ impression) / CVR AUC (subset đã click) | ~80 triệu samples |
| **KuaiRand** (bonus) | Short-video feed, 12 tín hiệu feedback + randomized exposure | NDCG@10 / Recall@50, click = positive | ~chục triệu interactions |

Link: AliCCP https://tianchi.aliyun.com/dataset/408 · KuaiRand https://kuairand.com
Baseline chính thức: NISE (AliCCP), CWM (KuaiRand).

## Luật
- **In scope:** mọi thư viện open-source, mọi paper / public solution / pretrained weights, sửa bất kỳ stage nào
- **Out of scope:** ❌ external training data; ❌ pretrained weights train trên test label của 2 benchmark này; ❌ đụng hidden test khi dev
- Compute budget: TBD. Hidden test chấm **một lần duy nhất** trên final submission

## Chấm điểm
| Tiêu chí | Trọng số | Nội dung |
|---|---|---|
| Technical Execution | 35% | **Primary metric**: `delta(m) = score_agent(m) − score_baseline(m)`, lấy trung bình các metric. Chấm ở **điểm converged** (val score không cải thiện quá ε trong N vòng, hoặc hết budget), dùng validation-best checkpoint. + **Robustness**: fail thì recover/retry/route around được, không crash/stall/diverge |
| Innovation & Problem Insight | 20% | Agent chọn cải tiến gì và **vì sao** — không chấm implementation |
| Impact & Relevance | 20% | **Autonomy** — đo bằng **số lần can thiệp thủ công**, càng ít càng cao |
| Feasibility & Practicality | 15% | **Resource**: tổng token (in+out) + tổng **GPU-hours** để đạt converged |
| Presentation & Communication | 10% | Final Event only |

## Deliverables riêng
Ngoài Devpost + repo: **run/iteration logs** (mỗi vòng: hypothesis, code diff, metric, error & recovery),
số lần can thiệp thủ công, final predictions/checkpoint theo schema, bảng kết quả + delta so với baseline,
và báo cáo token + GPU-hours.

> Có Appendix A: primer về recommender system (pipeline recall→pre-rank→rank→re-rank, CTR/CVR, multi-task, AUC/NDCG/Recall, embedding & feature crossing) — đọc 1–2 tiếng là đủ nền.
> Tool gợi ý: LLM coding agent bất kỳ, hoặc **Trae** của ByteDance (trial 7 ngày).

---

# Track 3 — Implement a GPU Kernel for a Transformer Layer

## Đề bài
Tối ưu runtime của một Transformer layer trên GPU bằng phương pháp **AI-assisted**.
Nộp một hoặc nhiều GPU kernel implement layer đó và pass test case.

- Test case viết bằng **PyTorch hoặc TensorFlow** (chọn 1 là đủ); được sửa implementation của layer → tự quyết fuse phần nào vào 1 kernel
- **Sai số cho phép: relative error < 0.02, absolute error < 0.002** so với bản gốc
- Test nhiều shape khác nhau (batch size, seq length, dimension lớn/nhỏ) — được viết shape check để chọn implementation khác nhau cho từng shape; **mọi tổ hợp shape sẽ được công bố trước**
- Tự chạy & tối ưu **trên máy của mình** (GPU của mình) → optimization phụ thuộc card
- Khuyến khích mạnh dùng AI tools; **tech report rõ ràng về AI skills/tools đã dùng → được điểm bonus**

## Hướng tối ưu gợi ý
Operator fusion · memory layout · reduced precision · tensor core · softmax optimization ·
custom CUDA / Triton / TF / PyTorch implementation.

## Scope
In scope: AI-based code generation, GPU kernel fusion, dùng profiling tools.
Out of scope: production-ready deployment.

## Việc cần làm
1. Tải benchmark script (`torch_transformer_benchmark.py` **hoặc** `tensorflow_transformer_benchmark.py`)
2. Implement phần customized-implementation, tối ưu hết mức
3. Chạy trên máy mình
4. Viết tech report: môi trường (CPU/GPU/DISK), các optimization đã làm, kết quả test cuối

Chấm điểm: bộ tiêu chí chung 35/20/20/15/10.

---

# Track 4 — Shopping Copilot: AI Conversational Search & Recommendations

## Đề bài
Xây shopping agent hội thoại trên dataset Amazon, vượt qua keyword matching tĩnh.
Bốn trụ cột bắt buộc:

**I. Core Architecture — Intent Routing & Hybrid Pipeline**
- **Dual-track routing**: phát hiện intent → track filter chính xác cao cho "Buying" (khoá hard constraint) / track dense retrieval đa dạng cho "Browsing" (cross-category)
- **Pipeline**: in-memory, "Multi-Route Retrieval → LLM Semantic Ranking" (keyword + category + vector similarity)

**II. Dialog Strategy — Multi-Turn Scenario Evolution**
- **Dynamic state machine**: xử lý Information Accumulation (slot tăng dần) và Intent Override (xoá & ghi lại slot)
- **Proactive guidance**: khi Over-Generality (pool ứng viên quá lớn) → cắt retrieval, chủ động hỏi clarification có cấu trúc

**III. Self-Evolution — Dynamic Context Programming**
- **Runtime adaptation**: personalized context distillation từ lịch sử hội thoại, cập nhật short-term session state + long-term user profile
- **Adaptive orchestration**: re-orchestrate workflow lúc runtime, tự tinh chỉnh logic guidance

**IV. Evaluation Matrix** (neo vào record mua hàng thực trong dataset)
- **Coverage**: Hit Rate@K (giai đoạn retrieval)
- **Precision**: MRR / Top-K Hit Rate (LLM đẩy đúng món đã mua lên top)
- **Efficiency**: **MTTC (Mean Turns to Conversion)** — ít lượt hơn = điểm cao hơn

## Constraints
- ❌ UI/UX (chấm hoàn toàn qua backend API + headless pipeline)
- ❌ Train / full fine-tune base LLM
- ❌ Vector DB cluster ngoài — **phải chạy hoàn toàn in-memory**
- ❌ Multi-modal — chỉ text catalog, structured metadata, text dialog
- ⚠️ **Hard limit 10 turns/session — vượt là bị terminate và 0 điểm**
- ⚠️ Catalog read-only, cấm mutate hoặc inject ASIN giả
- Giả định: input đã sạch (không cần xử lý typo/ASR noise), catalog tĩnh, mỗi session 1 user

## Tài nguyên
- Catalog đóng băng **50.000 sản phẩm** từ Amazon Reviews 2023 (Clothing_Shoes_and_Jewelry)
- **200 session public** để dev + **800 session private** để chấm cuối (user & target product tách biệt)
- Starter agent BM25 yếu, **local evaluator deterministic** (Hit Rate@10, MRR, MTTC, Efficiency, TechnicalScore), Python agent interface + API contract, SHA256 checksum
- Repo: https://github.com/TechJam2026/techjam-conversational-search
  (Kit release: `/releases/tag/participant-kit`) · Data gốc: https://amazon-reviews-2023.github.io/
- ⚠️ **BTC KHÔNG cấp API key / model access / credit** — dùng LLM ngoài thì tự trả tiền, và **không được commit secret**. Không bắt buộc dùng LLM trả phí.

Chấm điểm: bộ tiêu chí chung 35/20/20/15/10.

---

# Track 5 — Robust Detection of AI-Generated Images

## Đề bài
Phân biệt ảnh AI-generated vs ảnh thật, **giữ được độ chính xác sau khi ảnh bị hậu xử lý / phát tán lại**.
Không chỉ accuracy trên data sạch — phải bàn rõ trade-off giữa robustness, generalisation và false positive.

## Các phép biến đổi phải chịu được
| Transform | Tham số | Tình huống thực tế |
|---|---|---|
| JPEG Compression | quality 90 / 70 / 50 / 30 | Re-encode khi lên MXH, nhắn tin |
| Gaussian Blur | σ = 0.5 / 1.0 / 2.0 | Out-of-focus |
| Resize | scale 0.5× / 0.25× rồi upscale lại | Sinh thumbnail |
| Gaussian Noise | σ = 0.02 / 0.05 / 0.10 | Nhiễu cảm biến thiếu sáng |
| Color Jitter | brightness/contrast/saturation ±20% | App filter, auto-enhance |
| Center Crop | crop 80% | Cắt ảnh đại diện |

## Constraints
- In scope: AIGC detection mức ảnh, robustness, feature engineering, model design, evaluation design, error analysis, explainability
- Out of scope: production deployment, hệ thống moderation toàn platform, video/audio
- ⚠️ **Model phải < 2B parameters**; quy mô prototype hackathon

## Dataset
- https://huggingface.co/datasets/saberzl/SID_Set
- https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- https://modelscope.cn/datasets/hy2628982280/WildFake/summary
- **Validation set demo (KHÔNG dùng để train, không tính điểm):** subset WildFake — Non-AIGC COCO val2017 (4998 ảnh), AIGC DALL·E Advanced (8843 ảnh)

## Deliverables riêng
Ngoài Devpost + repo + demo video:
- **Script nhận vào 1 thư mục ảnh, xuất JSON gồm `image_path` và `pred`** (confidence là AIGC)
- **Robustness evaluation summary**: bảng/biểu đồ so sánh clean vs transformed
- **Error analysis note**: false positive / false negative tiêu biểu và trade-off

Chấm điểm: bộ tiêu chí chung 35/20/20/15/10.
