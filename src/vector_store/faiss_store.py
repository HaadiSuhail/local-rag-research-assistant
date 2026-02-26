import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict


class FaissVectorStore:
    def __init__(self, embedding_dim: int):
        """
        Simple FAISS inner-product index.
        Assumes embeddings are already normalized if cosine similarity is desired.
        """
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata: List[Dict] = []

    def add(self, embeddings: np.ndarray, metadata: List[Dict]):
        """
        Adds embeddings to index.
        embeddings: np.ndarray of shape (n, dim)
        metadata: list of dicts of same length
        """
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings must be a numpy array.")

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Searches the index using a single query embedding.
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "score": float(score),
                **self.metadata[idx]
            })

        return results
    
    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))

        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

    @classmethod
    def load(cls, path: Path):
        index = faiss.read_index(str(path / "index.faiss"))

        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        obj = cls(index.d)
        obj.index = index
        obj.metadata = metadata
        return obj

    def __len__(self):
        return self.index.ntotal