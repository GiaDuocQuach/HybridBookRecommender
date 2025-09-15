# BookRec — cấu trúc lại (giữ logic, bổ sung đánh giá)
## Local
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python scripts/eda_runner.py
python scripts/train_ranker.py
python scripts/eval_ranking.py
export PYTHONPATH=./src
export FLASK_APP=src/bookrec/api/app.py
export SECRET_KEY=change_me
flask run -h 127.0.0.1 -p 8080
## Docker
docker build -t bookrec:latest .
docker run --rm -it -p 8080:8080 -e FLASK_APP=src/bookrec/api/app.py -e PYTHONPATH=/app/src -e SECRET_KEY=change_me -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data -v $(pwd)/reports:/app/reports bookrec:latest
## Eval (offline)
python scripts/eval_ranking.py  # -> reports/ranking_metrics_by_category.csv
## Test
PYTHONPATH=./src pytest -q

## Hướng dẫn chạy trên Windows (PowerShell)

> Mở PowerShell tại thư mục **gốc** của dự án (nơi có `requirements.txt`, `src`, `scripts`, `web`)

```powershell
# 1) Tạo venv & cài dependencies
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2) Cho Python nhìn thấy 'src'
$env:PYTHONPATH = (Resolve-Path .\src).Path

# 3) EDA + chuẩn bị dữ liệu (gọi đúng các hàm trong eda.py của bạn)
python .\scripts\eda_runner.py

# 4) Huấn luyện LightGBM + xuất biểu đồ phân tích
python .\scripts	rain_ranker.py

# 5) (Tuỳ chọn) Đánh giá ranking offline theo category
python .\scripts\eval_ranking.py

# 6) Chạy Flask app
python -m flask --app src/bookrec/api/app.py run -h 127.0.0.1 -p 8080
```
