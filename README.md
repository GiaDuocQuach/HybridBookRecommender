# README — Hướng dẫn chạy (Runbook)

Dự án gợi ý sách kết hợp **semantic search + lexical + LightGBM reranker**, giao diện Flask.

---

## 1) Chuẩn bị môi trường

* **Python**: 3.10+
* Khuyên dùng **virtualenv** (venv).

### Tạo venv

**Windows (PowerShell)**

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Cài thư viện

```bash
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2) Cấu trúc thư mục tối thiểu

```
project_root/
├─ data/
│  └─ books_data_clean.csv              # dữ liệu sạch
├─ models/                              # sẽ sinh trong quá trình chạy
├─ web/
│  ├─ templates/                        # index.html, login.html, (detail.html nếu có)
│  └─ static/                           # style.css, app.js
├─ app.py
├─ eda.py, eda_runner.py
├─ train_model.py, train_ranker.py
├─ eval_ranking.py
├─ embedding.py, ranker.py, metrics.py
└─ README.md
```

---

## 3) Thiết lập biến môi trường (khuyến nghị)

Tạo file `.env` (hoặc export biến trước khi chạy):

```
SECRET_KEY=replace_with_a_secret_key_production
INTERACTIONS_DB=interactions.db
SENTENCE_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBED_DEVICE=cpu
# Trọng số trộn điểm
ALPHA_LGB=0.55
BETA_SESS=0.20
GAMMA_USER=0.10
DELTA_QUERY=0.12
EPS_LEX=0.08
```

---

## 4) Chuẩn bị dữ liệu

Nếu **đã có** `data/books_data_clean.csv` thì bỏ qua bước này.

Nếu **chưa có**:

* Dùng `clean_data.py` để hợp nhất/chuẩn hoá từ nguồn Tiki & Fahasa (cần dữ liệu raw).
* Hoặc dùng file mẫu bạn có sẵn.

---

## 5) Tiền xử lý & chia tập (EDA + split)

```bash
python eda_runner.py
```

Sinh các file:

* `models/le_cat.pkl`, `models/le_pop.pkl`
* `data/X_train.npy`, `data/y_train.npy`, `data/X_val.npy`, `data/y_val.npy`, ...
* `reports/figures/*` (biểu đồ EDA)

---

## 6) Train ranker (LightGBM) + chọn threshold tối ưu

```bash
python train_ranker.py
```

Kết quả:

* `models/book_ranker.pkl`
* `models/optimal_threshold.txt`
* Biểu đồ: feature importance, PR-curve, confusion matrix (trong `reports/figures/` nếu script có vẽ).

---

## 7) (Tuỳ chọn) Đánh giá xếp hạng

```bash
python eval_ranking.py
```

Sinh `reports/ranking_metrics_by_category.csv` với NDCG/Recall/HitRate/MRR theo từng thể loại.

> Lưu ý nhất quán đặc trưng: `price_log` nên là `log1p(price_k)` ở mọi nơi (train/eval/app).

---

## 8) Chạy web app

```bash
python app.py
```

* Mặc định chạy tại **[http://127.0.0.1:8080](http://127.0.0.1:8080)**.
* Lần đầu có thể mất thời gian để **tính embeddings** và lưu vào `models/book_embeddings.npy`.
* Đăng nhập (username tuỳ ý) → vào trang tìm kiếm → gõ từ khoá/chọn bộ lọc → nhận **TOP‑9** gợi ý.

**Các endpoint chính** (tham khảo):

* `POST /recommend` – trả về danh sách sách đã rerank.
* `POST /click`     – ghi lại click vào SQLite.
* `GET  /book/<id>` – trang chi tiết.
* `GET|POST /login`, `GET /logout`.
