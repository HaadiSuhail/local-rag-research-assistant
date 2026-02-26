from typing import List, Dict, Optional
import numpy as np


class Retriever:
    def __init__(
        self,
        embedder,
        vector_store,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ):
        """
        Retrieval abstraction layer.
        
        embedder: HFEmbedder instance
        vector_store: FaissVectorStore instance
        top_k: number of results to retrieve
        score_threshold: optional minimum similarity score
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieves top-k relevant chunks for a query.
        """
        # 1. Embed query
        query_embedding = self.embedder.encode([query])

        # 2. Search vector store
        results = self.vector_store.search(query_embedding, top_k=self.top_k)

        # 3. Optional filtering
        if self.score_threshold is not None:
            results = [
                r for r in results
                if r["score"] >= self.score_threshold
            ]

        return results