import os
import re
import unicodedata
import sqlite3
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template, session, redirect, url_for

from bookrec.models.embedding import EmbeddingModel
from bookrec.models.ranker import BookRanker

app = Flask(
    __name__,
    template_folder='../../../web/templates',
    static_folder='../../../web/static'
)
app.secret_key = os.getenv('SECRET_KEY', 'replace_with_a_secret_key_production')

DATA_CSV   = os.path.join('data', 'books_data_clean.csv')
EMB_NPY    = os.path.join('models', 'book_embeddings.npy')
RANKER_PKL = os.path.join('models', 'book_ranker.pkl')
LE_CAT_PKL = os.path.join('models', 'le_cat.pkl')
LE_POP_PKL = os.path.join('models', 'le_pop.pkl')
THR_TXT    = os.path.join('models', 'optimal_threshold.txt')
DB_PATH    = os.getenv('INTERACTIONS_DB', 'interactions.db')

def _get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    con = _get_conn()
    try:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                book_idx INTEGER NOT NULL,
                action TEXT DEFAULT 'click',
                ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user_ts ON interactions(user_id, ts)")
        con.commit()
    finally:
        con.close()

try:
    init_db()
except Exception as e:
    print(f"[WARN] init_db at import time failed: {e}")

def save_interaction(user_id: str, book_idx: int, action: str = 'click'):
    try:
        init_db()
    except Exception:
        pass
    con = _get_conn()
    try:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO interactions(user_id, book_idx, action) VALUES (?, ?, ?)",
            (user_id, int(book_idx), action)
        )
        con.commit()
    finally:
        con.close()

def get_user_history_longterm(user_id: str, limit: Optional[int] = None) -> List[int]:
    try:
        init_db()
    except Exception:
        pass
    con = _get_conn()
    try:
        cur = con.cursor()
        if limit:
            cur.execute(
                "SELECT book_idx FROM interactions WHERE user_id=? ORDER BY ts DESC LIMIT ?",
                (user_id, int(limit))
            )
        else:
            cur.execute("SELECT book_idx FROM interactions WHERE user_id=? ORDER BY ts DESC", (user_id,))
        return [int(r[0]) for r in cur.fetchall()]
    finally:
        con.close()

if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(f"Không tìm thấy {DATA_CSV}. Hãy chạy scripts/eda_runner.py trước.")

df: pd.DataFrame = pd.read_csv(DATA_CSV).reset_index(drop=True)
df['book_idx'] = df.index

for col in ['price_k', 'rating', 'total_sales']:
    df[col] = pd.to_numeric(df.get(col, 0.0), errors='coerce').fillna(0.0)

def map_popularity_ui(x: str) -> str:
    if x == 'Trending':
        return 'Trending'
    if x in ['Warm', 'Hot']:
        return 'Popular'
    return 'Unpopular'

if 'popularity_level' in df.columns:
    df['pop_mapped'] = df['popularity_level'].astype(str).apply(map_popularity_ui)
else:
    df['pop_mapped'] = 'Unpopular'

df['rating_norm'] = df['rating'] / 5.0
df['sales_pct']   = df.groupby('category')['total_sales'].rank(method='average', pct=True)
df['price_log']   = np.log1p(df['price_k'])

df['clean_desc']   = df.get('clean_desc', '')
df['text_for_emb'] = df['title'].astype(str) + '. ' + df['clean_desc'].astype(str)

def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize('NFD', s or '')
    return ''.join(ch for ch in nfkd if not unicodedata.combining(ch))

def normalize_text(s: str) -> str:
    s = strip_accents((s or '').lower())
    s = re.sub(r'[^0-9a-z\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

df['title_norm']    = df['title'].astype(str).apply(normalize_text)
df['category_norm'] = df['category'].astype(str).apply(normalize_text)
df['desc_norm']     = df['clean_desc'].astype(str).str.slice(0, 200).apply(normalize_text)

SENTENCE_MODEL = os.getenv('SENTENCE_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
DEVICE         = os.getenv('EMBED_DEVICE', 'cpu')

embedding_model = EmbeddingModel(
    model_name=SENTENCE_MODEL,
    emb_path=EMB_NPY,
    texts=df['text_for_emb'].tolist(),
    device=DEVICE
)
book_embeddings = embedding_model.embeddings  # normalized

ranker = BookRanker(RANKER_PKL)
le_cat = joblib.load(LE_CAT_PKL) if os.path.exists(LE_CAT_PKL) else None
le_pop = joblib.load(LE_POP_PKL) if os.path.exists(LE_POP_PKL) else None

OPTIMAL_THRESHOLD = 0.5
if os.path.exists(THR_TXT):
    try:
        with open(THR_TXT, 'r', encoding='utf-8') as f:
            OPTIMAL_THRESHOLD = float(f.read().strip() or '0.5')
    except Exception:
        OPTIMAL_THRESHOLD = 0.5

ALPHA = float(os.getenv("ALPHA_LGB",  "0.55"))  # ranker
BETA  = float(os.getenv("BETA_SESS", "0.20"))  # sim_session
GAMMA = float(os.getenv("GAMMA_USER","0.10"))  # sim_user
DELTA = float(os.getenv("DELTA_QUERY","0.12")) # embedding
EPS_LEX = float(os.getenv("EPS_LEX",  "0.08")) # phrase + token

def _pop_for_encoder(rows: pd.DataFrame) -> pd.Series:
    if 'popularity_level' in rows.columns:
        return rows['popularity_level'].astype(str).fillna('Cold')
    rev = {'Unpopular': 'Cold', 'Popular': 'Warm', 'Trending': 'Trending'}
    return rows['pop_mapped'].astype(str).map(rev).fillna('Cold')

def build_features(df_subset: pd.DataFrame) -> np.ndarray:
    # category enc
    if le_cat is not None:
        cat_enc = le_cat.transform(df_subset['category'].astype(str).fillna(''))
    else:
        cat_enc = df_subset['category'].astype(str).fillna('').apply(lambda s: abs(hash(s)) % 1000).to_numpy()

    if le_pop is not None:
        pop_raw = _pop_for_encoder(df_subset)
        pop_enc = le_pop.transform(pop_raw)
    else:
        pop_src = df_subset['pop_mapped'].astype(str).fillna('Unpopular')
        pop_enc = pop_src.apply(lambda s: abs(hash(s)) % 100).to_numpy()
    price_log   = df_subset['price_log'].astype(float).to_numpy()
    rating_norm = df_subset['rating_norm'].astype(float).to_numpy()
    sales_pct   = df_subset['sales_pct'].astype(float).to_numpy()
    X = np.vstack([cat_enc, pop_enc, price_log, rating_norm, sales_pct]).T.astype(float)
    return X

def compute_session_embedding() -> Optional[np.ndarray]:
    hist = session.get('history', [])
    if not hist:
        return None
    valid = [book_embeddings[i] for i in hist if 0 <= i < len(book_embeddings)]
    if not valid:
        return None
    v = np.mean(valid, axis=0)
    n = np.linalg.norm(v)
    return v / (n + 1e-12) if n > 0 else None

def compute_user_embedding_longterm(user_id: str) -> Optional[np.ndarray]:
    idxs = get_user_history_longterm(user_id)
    if not idxs:
        return None
    valid = [book_embeddings[i] for i in idxs if 0 <= i < len(book_embeddings)]
    if not valid:
        return None
    v = np.mean(valid, axis=0)
    n = np.linalg.norm(v)
    return v / (n + 1e-12) if n > 0 else None

def _fmt_price_vnd(price_k: float) -> str:
    try:
        vnd = int(round(float(price_k) * 1000.0))
    except Exception:
        vnd = 0
    return f"{vnd:,}đ".replace(",", ".")

def _tokenize_lower(s: str) -> List[str]:
    return re.findall(r"[0-9a-zA-ZÀ-ỹ]+", (s or "").lower())

def _lexical_scores_for_indices(idx_list: List[int], keyword: str) -> np.ndarray:
    """Tính điểm lexical (phrase + token overlap) cho danh sách index sách."""
    q_norm = normalize_text(keyword)
    if not q_norm:
        return np.zeros(len(idx_list), dtype=float)
    q_tokens = set(q_norm.split())

    scores = []
    for i in idx_list:
        t = df.at[i, 'title_norm']
        c = df.at[i, 'category_norm']
        d = df.at[i, 'desc_norm']

        phrase = 0.0
        if q_norm in t: phrase += 1.0
        if q_norm in c: phrase += 0.6
        if q_norm in d: phrase += 0.3
        phrase *= 0.7

        tset = set(t.split()); cset = set(c.split())
        if q_tokens:
            inter_t = len(tset & q_tokens) / len(q_tokens)
            inter_c = len(cset & q_tokens) / len(q_tokens)
        else:
            inter_t = inter_c = 0.0
        token_score = 0.7 * inter_t + 0.3 * inter_c
        token_score *= 0.3

        scores.append(phrase + token_score)

    return np.asarray(scores, dtype=float)


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        if not username:
            return render_template('login.html', error='Vui lòng nhập username')
        session.clear()
        session['user_id'] = username
        session['history'] = []
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    categories = sorted(df['category'].dropna().astype(str).unique())
    if 'price_range' in df.columns:
        price_ranges = sorted(
            df['price_range'].dropna().astype(str).unique().tolist(),
            key=lambda s: (s.startswith('>'), s)
        )
    else:
        price_ranges = ['<100k','100-200k','200-500k','>500k']
    pop_levels  = ['Popular', 'Unpopular', 'Trending']
    rating_bins = ['0-3', '3-5']
    return render_template(
        'index.html',
        categories=categories,
        price_ranges=price_ranges,
        pop_levels=pop_levels,
        rating_bins=rating_bins,
        username=session.get('user_id')
    )

@app.route('/click', methods=['POST'])
def click():
    if 'user_id' not in session:
        return jsonify({'ok': False, 'error': 'not_logged_in'}), 401
    data = request.get_json(force=True, silent=True) or {}
    book_idx = int(data.get('book_idx', -1))
    if not (0 <= book_idx < len(df)):
        return jsonify({'ok': False, 'error': 'invalid_book_idx'}), 400
    hist = session.get('history', [])
    hist = [book_idx] + hist[:49]
    session['history'] = hist
    save_interaction(session['user_id'], book_idx, 'click')
    return jsonify({'ok': True})

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'user_id' not in session:
        return jsonify({'error': 'Chưa login'}), 401

    payload = request.get_json(force=True) or {}
    cat         = (payload.get('category') or '').strip()
    prices      = payload.get('price_ranges', []) or []
    pop_levels  = payload.get('pop_levels', []) or []
    rating_bins = payload.get('rating_bins', []) or []
    keyword     = (payload.get('keyword') or '').strip()
    TOPN = int(payload.get('top_k', 9)) or 9

    df_filtered = df
    if cat:
        df_filtered = df_filtered[df_filtered['category'].astype(str) == cat]
    if prices and 'price_range' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['price_range'].astype(str).isin(prices)]
    if pop_levels:
        df_filtered = df_filtered[df_filtered['pop_mapped'].astype(str).isin(pop_levels)]
    if rating_bins:
        mask = pd.Series(False, index=df_filtered.index)
        for b in rating_bins:
            if b == '0-3':
                mask |= (df_filtered['rating'] < 3)
            else:
                mask |= (df_filtered['rating'] >= 3)
        df_filtered = df_filtered[mask]

    if keyword:
        q_vec = embedding_model.encode([keyword])[0]
        sims_all = book_embeddings.dot(q_vec)  # (N,)
        order = np.argsort(-sims_all)

        filt_set = set(df_filtered.index)
        cand_idx = [i for i in order if i in filt_set][:300]

        # thiếu thì nới lỏng (giữ liên quan truy vấn)
        seen = set(cand_idx)
        if len(cand_idx) < 300:
            for i in order:
                if i in seen:
                    continue
                cand_idx.append(i)
                seen.add(i)
                if len(cand_idx) >= 300:
                    break

        candidates = df.loc[cand_idx].copy()
        candidates['sim_query_raw'] = sims_all[cand_idx]
    else:
        candidates = (df_filtered if not df_filtered.empty else df).copy()
        candidates = candidates.sort_values(['total_sales', 'rating'], ascending=False).head(400).copy()
        candidates['sim_query_raw'] = 0.0

    if len(candidates) > 0:
        s = candidates['sim_query_raw'].values.astype(float)
        s_min, s_max = float(np.min(s)), float(np.max(s))
        s_norm = (s - s_min) / (s_max - s_min + 1e-12)
        candidates['sim_query'] = s_norm
    else:
        candidates['sim_query'] = 0.0

    orig_idx_list = candidates.index.tolist()
    candidates = candidates.reset_index().rename(columns={'index': 'orig_idx'})
    candidates['lexical'] = _lexical_scores_for_indices(candidates['orig_idx'].astype(int).tolist(), keyword)

    X = build_features(candidates)
    probs = np.asarray(ranker.predict_proba(X))
    if probs.ndim == 2 and probs.shape[1] > 1:
        probs = probs[:, 1]
    lightgbm_score = (probs > OPTIMAL_THRESHOLD).astype(float)
    candidates['lightgbm_score'] = lightgbm_score

    session_emb = compute_session_embedding()
    user_emb    = compute_user_embedding_longterm(session['user_id'])

    sim_s = []
    sim_u = []
    for i in candidates['orig_idx'].astype(int).tolist():
        sim_s.append(float(book_embeddings[i].dot(session_emb)) if session_emb is not None else 0.0)
        sim_u.append(float(book_embeddings[i].dot(user_emb))    if user_emb    is not None else 0.0)
    candidates['sim_session'] = sim_s
    candidates['sim_user']    = sim_u

    candidates['final_score'] = (
        ALPHA * candidates['lightgbm_score']
        + BETA  * candidates['sim_session']
        + GAMMA * candidates['sim_user']
        + DELTA * candidates['sim_query']
        + EPS_LEX * candidates['lexical']
    )

    top = candidates.sort_values('final_score', ascending=False).head(TOPN)
    need = TOPN - len(top)
    if need > 0:
        pool = df.drop(index=top['orig_idx'].astype(int).tolist(), errors='ignore').copy()
        pool = pool.sort_values(['total_sales','rating'], ascending=False).head(600).copy()

        pool['sim_query_raw'] = 0.0
        if keyword:
            q_vec = embedding_model.encode([keyword])[0]
            pool_idx = pool.index.to_numpy()
            pool['sim_query_raw'] = book_embeddings[pool_idx].dot(q_vec)
        s = pool['sim_query_raw'].values.astype(float)
        s_min, s_max = float(np.min(s)), float(np.max(s))
        pool['sim_query'] = (s - s_min) / (s_max - s_min + 1e-12)

        pool = pool.reset_index().rename(columns={'index':'orig_idx'})
        pool['lexical'] = _lexical_scores_for_indices(pool['orig_idx'].astype(int).tolist(), keyword)

        X2 = build_features(pool)
        probs2 = np.asarray(ranker.predict_proba(X2))
        if probs2.ndim == 2 and probs2.shape[1] > 1:
            probs2 = probs2[:,1]
        lb2 = (probs2 > OPTIMAL_THRESHOLD).astype(float)
        pool['lightgbm_score'] = lb2

        sim_s2, sim_u2 = [], []
        for i in pool['orig_idx'].astype(int).tolist():
            sim_s2.append(float(book_embeddings[i].dot(session_emb)) if session_emb is not None else 0.0)
            sim_u2.append(float(book_embeddings[i].dot(user_emb))    if user_emb    is not None else 0.0)
        pool['sim_session'] = sim_s2
        pool['sim_user']    = sim_u2

        pool['final_score'] = (
            ALPHA * pool['lightgbm_score']
            + BETA  * pool['sim_session']
            + GAMMA * pool['sim_user']
            + DELTA * pool['sim_query']
            + EPS_LEX * pool['lexical']
        )
        fillers = pool.sort_values('final_score', ascending=False).head(need)
        top = pd.concat([top, fillers], ignore_index=True)

    out = []
    for _, r in top.head(TOPN).iterrows():
        i = int(r['orig_idx'])
        stars_full = int(round(df.at[i, 'rating']))
        stars = '★' * stars_full + '☆' * (5 - stars_full)
        out.append({
            'book_idx': i,
            'title': str(df.at[i, 'title']),
            'author': str(df.at[i, 'author']),
            'price': _fmt_price_vnd(df.at[i, 'price_k']),
            'sold_count': int(df.at[i, 'total_sales']),
            'rating': float(df.at[i, 'rating']),
            'stars': stars,
            'image_url': str(df.at[i, 'image']),
            'book_url': str(df.at[i, 'url']),
            'popularity_level': str(df.at[i, 'pop_mapped']),
            'final_score': float(r['final_score']),
        })
    return jsonify({'results': out})

@app.route('/book/<int:book_idx>')
def book_detail(book_idx: int):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if not (0 <= book_idx < len(df)):
        return 'Không tìm thấy sách', 404
    rating_val = float(df.at[book_idx, 'rating'])
    stars = '★' * int(round(rating_val)) + '☆' * (5 - int(round(rating_val)))
    return render_template(
        'detail.html',
        title=str(df.at[book_idx, 'title']),
        author=str(df.at[book_idx, 'author']),
        category=str(df.at[book_idx, 'category']),
        popularity_level=str(df.at[book_idx, 'pop_mapped']),
        rating=rating_val,
        stars=stars,
        price_k=float(df.at[book_idx, 'price_k']),
        sold_count=int(df.at[book_idx, 'total_sales']),
        image_url=str(df.at[book_idx, 'image']),
        book_url=str(df.at[book_idx, 'url']),
        description=str(df.get('description', pd.Series([''] * len(df))).iat[book_idx]),
        username=session.get('user_id')
    )

if __name__ == '__main__':
    try:
        init_db()
    except Exception as e:
        print(f'[WARN] init_db on __main__: {e}')
    app.run(host='127.0.0.1', port=8080, debug=False)
