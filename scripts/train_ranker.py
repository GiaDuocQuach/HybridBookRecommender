from bookrec.models.train_model import train_and_evaluate

if __name__ == "__main__":
    thr = train_and_evaluate()
    # Lưu ngưỡng tối ưu để Flask app có thể đọc (nếu cần)
    import os
    os.makedirs("models", exist_ok=True)
    with open("models/optimal_threshold.txt", "w", encoding="utf-8") as f:
        f.write(str(thr))
    print("ĐÃ LƯU: models/optimal_threshold.txt =", thr)
