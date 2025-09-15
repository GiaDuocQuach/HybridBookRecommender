import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    data_path = os.path.join('data', 'books_data_clean.csv')
    df = pd.read_csv(data_path)

    df['sales_pct'] = df.groupby('category')['total_sales'].rank(pct=True)

    def map_popularity(x):
        if x == 'Trending':
            return 'Trending'
        if x in ['Warm', 'Hot']:
            return 'Popular'
        return 'Unpopular'

    df['pop_mapped'] = df['popularity_level'].apply(map_popularity)

    alpha = 0.7
    beta = 0.3
    df['relevance_score'] = alpha * df['sales_pct'] + beta * (df['rating'] / 5.0)

    pct = 0.30
    df['target'] = df.groupby('category')['relevance_score'].transform(
        lambda x: (x >= x.quantile(1 - pct)).astype(int)
    )

    return df


def perform_eda(df):
    plt.figure(figsize=(10, 6))
    target_counts = df['target'].value_counts()
    plt.pie(target_counts, labels=['Trung bình', 'Đáng mua'], autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'], startangle=90)
    plt.savefig('reports/figures/target_distribution.png')
    plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    sns.histplot(df['price_k'], kde=True, ax=axs[0, 0], bins=30)
    axs[0, 0].set_title('Phân phối giá sách (nghìn VND)')
    axs[0, 0].set_xlabel('Giá (nghìn VND)')

    sns.boxplot(x='target', y='rating', data=df, ax=axs[0, 1])
    axs[0, 1].set_title('Phân phối rating theo lớp mục tiêu')

    current_ticks = axs[0, 1].get_xticks()
    if len(current_ticks) == 2:
        axs[0, 1].set_xticks(current_ticks)
        axs[0, 1].set_xticklabels(['Không phổ biến', 'Phổ biến'])

    sample_size = min(1000, len(df))
    sns.scatterplot(x='total_sales', y='rating', hue='target', data=df.sample(sample_size), ax=axs[1, 0], palette={0: 'red', 1: 'blue'}, alpha=0.6)
    axs[1, 0].set_title('Tương quan doanh số và rating')
    axs[1, 0].set_xscale('log')

    top_cats = df['category'].value_counts().nlargest(10).index
    df_top = df[df['category'].isin(top_cats)]
    cat_popularity = df_top.groupby('category')['target'].mean().sort_values(ascending=False)

    sns.barplot(x=cat_popularity.values, y=cat_popularity.index, ax=axs[1, 1], color='#1f77b4')
    axs[1, 1].set_title('Tỷ lệ sách phổ biến theo thể loại (Top 10)')
    axs[1, 1].set_xlabel('Tỷ lệ sách phổ biến')

    plt.tight_layout()
    plt.savefig('reports/figures/feature_distributions.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    corr_cols = ['price_k', 'rating', 'total_sales', 'sales_pct', 'target']
    available_cols = [col for col in corr_cols if col in df.columns]
    corr_matrix = df[available_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', annot_kws={"size": 12}, cbar=False)
    plt.title('Ma trận tương quan giữa các đặc trưng')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_matrix.png')
    plt.close()

    print("EDA hoàn thành! Biểu đồ đã lưu tại reports/figures")


def prepare_datasets(df):
    feature_cols = ['price_k', 'rating', 'total_sales', 'sales_pct', 'popularity_level']

    df['rating_norm'] = df['rating'] / 5.0
    df['price_log'] = np.log1p(df['price_k'])

    from sklearn.preprocessing import LabelEncoder

    popularity_encoder = LabelEncoder()
    df['pop_enc'] = popularity_encoder.fit_transform(df['popularity_level'])

    category_encoder = LabelEncoder()
    df['cat_enc'] = category_encoder.fit_transform(df['category'])

    final_feature_cols = ['cat_enc', 'pop_enc', 'price_log', 'rating_norm', 'sales_pct']
    X = df[final_feature_cols].values.astype(float)
    y = df['target'].values.astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    print(f"Kích thước datasets: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, category_encoder, popularity_encoder

if __name__ == '__main__':
    df = load_and_preprocess_data()
    perform_eda(df)
    X_train, X_val, X_test, y_train, y_val, y_test, cat_encoder, pop_encoder = prepare_datasets(df)

    np.save('data/X_train.npy', X_train)
    np.save('data/X_val.npy', X_val)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_val.npy', y_val)
    np.save('data/y_test.npy', y_test)

    joblib.dump(cat_encoder, 'models/le_cat.pkl')
    joblib.dump(pop_encoder, 'models/le_pop.pkl')

    print("Chuẩn bị dữ liệu hoàn tất!")
