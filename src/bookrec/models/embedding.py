import os
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModel:

    def __init__(self, model_name: str, emb_path: str, texts: list[str], device: str = 'cpu'):
        self.model = SentenceTransformer(model_name, device=device)
        self.emb_path = emb_path

        if os.path.exists(emb_path):
            self.embeddings = np.load(emb_path)
        else:
            self.embeddings = self.model.encode(
                texts, show_progress_bar=True, convert_to_numpy=True
            )
            os.makedirs(os.path.dirname(emb_path), exist_ok=True)
            np.save(emb_path, self.embeddings)
            print(f"Đã lưu embeddings vào {emb_path}")
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.embeddings = self.embeddings / norms

    def encode(self, texts: list[str]) -> np.ndarray:
        embs = self.model.encode(texts, convert_to_numpy=True)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embs / norms
