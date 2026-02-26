from typing import List, Optional
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import json


class HFEmbedder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Research-grade embedding wrapper.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.normalize = normalize

        print(f"🔹 Loading embedding model: {model_name}")
        print(f"🔹 Using device: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 16,
        show_progress_bar: bool = True
    ) -> np.ndarray:
        """
        Encodes a list of texts into embeddings.
        Returns numpy array of shape (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )

        return embeddings.astype("float32")

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[dict],
        save_dir: Path,
        name: str
    ):
        """
        Saves embeddings and metadata for reproducibility.
        """
        save_dir.mkdir(parents=True, exist_ok=True)

        np.save(save_dir / f"{name}_embeddings.npy", embeddings)

        with open(save_dir / f"{name}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"✔ Saved embeddings to {save_dir}")

    @staticmethod
    def load_embeddings(save_dir: Path, name: str):
        """
        Loads saved embeddings and metadata.
        """
        embeddings = np.load(save_dir / f"{name}_embeddings.npy")

        with open(save_dir / f"{name}_metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return embeddings, metadata