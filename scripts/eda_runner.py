import os, numpy as np, joblib
from bookrec.data.eda import load_and_preprocess_data, perform_eda, prepare_datasets

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    df = load_and_preprocess_data()
    perform_eda(df)
    X_train, X_val, X_test, y_train, y_val, y_test, cat_encoder, pop_encoder = prepare_datasets(df)

    np.save("data/X_train.npy", X_train)
    np.save("data/X_val.npy",   X_val)
    np.save("data/X_test.npy",  X_test)
    np.save("data/y_train.npy", y_train)
    np.save("data/y_val.npy",   y_val)
    np.save("data/y_test.npy",  y_test)

    joblib.dump(cat_encoder, "models/le_cat.pkl")
    joblib.dump(pop_encoder, "models/le_pop.pkl")
    print("EDA + chuẩn bị dữ liệu: DONE")
