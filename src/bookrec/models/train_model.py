import os
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


FEATURE_NAMES = ["category_enc", "pop_enc", "price_log", "rating_norm", "sales_pct"]


def plot_feature_importance(model, feature_names, save_path):

    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        return
    importance = np.asarray(importance)
    idx = np.argsort(importance)[::-1]
    names = np.array(feature_names)[idx]
    vals = importance[idx]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.title("Feature Importance")
    plt.barh(range(len(names)), vals[::-1])
    plt.yticks(range(len(names)), names[::-1])
    pad = (float(vals.max()) * 0.01) if vals.size and vals.max() > 0 else 0.1
    for i, v in enumerate(vals[::-1]):
        plt.text(v + pad, i, f"{v:.0f}", va="center")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):

    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    cmn = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _load_split_arrays():

    X_train = np.load("data/X_train.npy")
    X_val = np.load("data/X_val.npy")
    X_test = np.load("data/X_test.npy")
    y_train = np.load("data/y_train.npy")
    y_val = np.load("data/y_val.npy")
    y_test = np.load("data/y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_evaluate():

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = _load_split_arrays()

    clf = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        colsample_bytree=0.8,
        subsample=0.8,
        random_state=42,
    )

    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.log_evaluation(period=0)],  # không đổi logic, chỉ tắt log
    )

    joblib.dump(clf, "models/book_ranker.pkl")

    plot_feature_importance(
        clf, FEATURE_NAMES, "reports/figures/feature_importance.png"
    )

    y_proba_val = (
        clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_val)
    )
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba_val)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, label="LightGBM (val)")
    if 0 <= best_idx < len(recall):
        plt.scatter([recall[best_idx]], [precision[best_idx]], s=60)
        plt.text(
            recall[best_idx],
            precision[best_idx],
            f"thr={optimal_threshold:.3f}\nF1={f1_scores[best_idx]:.3f}",
            fontsize=9,
            ha="left",
            va="bottom",
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (Validation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/precision_recall_curve.png")
    plt.close()

    y_proba_test = (
        clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_test)
    )
    y_pred_test = (y_proba_test >= optimal_threshold).astype(int)
    plot_confusion_matrix(y_test, y_pred_test, "reports/figures/confusion_matrix.png")

    try:
        auc_test = roc_auc_score(y_test, y_proba_test)
        print(f"Test AUC: {auc_test:.4f}")
    except Exception:
        pass

    return optimal_threshold
