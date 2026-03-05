import numpy as np
from tqdm import tqdm
import json


class RAGEvaluator:

    def __init__(self, rag_system):
        self.rag = rag_system

    def evaluate(self, eval_path: str):

        with open(eval_path, "r") as f:
            dataset = json.load(f)

        recall_hits = 0
        reciprocal_ranks = []
        groundedness_scores = []

        for item in tqdm(dataset):
            question = item["question"]
            ground_truth = item["ground_truth_source"]

            docs = self.rag.retrieve(question)
            sources = [d.metadata.get("source", "") for d in docs]

            if ground_truth in sources:
                recall_hits += 1
                rank = sources.index(ground_truth) + 1
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)

            answer = self.rag.ask(question)

            context = " ".join([d.page_content for d in docs])
            groundedness_scores.append(self._groundedness(answer, context))

        return {
            "Recall@K": round(recall_hits / len(dataset), 3),
            "MRR": round(np.mean(reciprocal_ranks), 3),
            "Groundedness": round(np.mean(groundedness_scores), 3)
        }

    def _groundedness(self, answer, context):
        a = set(answer.lower().split())
        c = set(context.lower().split())
        if not a:
            return 0
        return len(a & c) / len(a)