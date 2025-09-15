import os, numpy as np, pandas as pd, joblib
from sklearn.preprocessing import LabelEncoder
from bookrec.evaluation.metrics import ndcg_at_k, recall_at_k, hit_rate_at_k, mrr_at_k
DATA_CSV = 'data/books_data_clean.csv'
LE_CAT = 'models/le_cat.pkl'; LE_POP = 'models/le_pop.pkl'; MODEL = 'models/book_ranker.pkl'
def ensure_encoders(df):
    if os.path.exists(LE_CAT) and os.path.exists(LE_POP):
        le_cat = joblib.load(LE_CAT); le_pop = joblib.load(LE_POP)
    else:
        le_cat = LabelEncoder().fit(df['category'].astype(str).fillna(''))
        pop_col = df['pop_mapped'] if 'pop_mapped' in df.columns else df.get('popularity_level', pd.Series(['Cold']*len(df)))
        le_pop = LabelEncoder().fit(pop_col.astype(str).fillna('Cold'))
        os.makedirs('models', exist_ok=True); joblib.dump(le_cat, LE_CAT); joblib.dump(le_pop, LE_POP)
    return le_cat, le_pop
def build_features(df, le_cat, le_pop):
    cat = le_cat.transform(df['category'].astype(str))
    pop_src = df['pop_mapped'] if 'pop_mapped' in df.columns else df.get('popularity_level', pd.Series(['Cold']*len(df)))
    pop = le_pop.transform(pop_src.astype(str))
    price_log = np.log1p(df.get('price_k', 0.0).astype(float) * 1000.0)
    rating_norm = df.get('rating', 0.0).astype(float) / 5.0
    sales_pct = df.get('sales_pct', pd.Series(np.zeros(len(df)))).astype(float)
    X = np.vstack([cat, pop, price_log, rating_norm, sales_pct]).T.astype(float)
    return X
def main():
    assert os.path.exists(DATA_CSV), f'Missing {DATA_CSV}. Hãy chạy scripts/eda_runner.py trước.'
    df = pd.read_csv(DATA_CSV)
    le_cat, le_pop = ensure_encoders(df)
    X = build_features(df, le_cat, le_pop)
    assert os.path.exists(MODEL), f'Missing {MODEL}. Hãy chạy scripts/train_ranker.py trước.'
    model = joblib.load(MODEL)
    scores = model.predict_proba(X)[:,1] if hasattr(model,'predict_proba') else model.predict(X)
    y = ((df.get('sales_pct', 0.0) > 0.6) | (df.get('rating', 0.0) >= 4.5)).astype(int).to_numpy()
    rows = []
    for cat, g in df.groupby('category', dropna=False):
        idx = g.index.to_numpy(); 
        if len(idx) < 3: continue
        y_true = y[idx]; y_score = scores[idx]
        rows.append({
            'category': str(cat), 'N': int(len(idx)),
            'NDCG@5': ndcg_at_k(y_true, y_score, k=5),
            'NDCG@10': ndcg_at_k(y_true, y_score, k=10),
            'Recall@20': recall_at_k(y_true, y_score, k=20),
            'HitRate@5': hit_rate_at_k(y_true, y_score, k=5),
            'MRR@10': mrr_at_k(y_true, y_score, k=10),
        })
    out = pd.DataFrame(rows).sort_values('N', ascending=False)
    os.makedirs('reports', exist_ok=True)
    out.to_csv('reports/ranking_metrics_by_category.csv', index=False, encoding='utf-8')
    print('Saved reports/ranking_metrics_by_category.csv'); print(out.head(10).to_string(index=False))
if __name__ == '__main__': main()
