from pathlib import Path

from src.utils.config_loader import load_config
from src.embeddings.hf_embedder import HFEmbedder
from src.vector_store.faiss_store import FaissVectorStore
from src.retrieval.retriever import Retriever
from src.generation.prompt_builder import PromptBuilder
from src.generation.hf_llm import HFLLM

import time

INDEX_DIR = Path("data/index")


def main():
    config = load_config()

    # ===== Load Components =====
    vector_store = FaissVectorStore.load(INDEX_DIR)

    embedder = HFEmbedder()

    llm = HFLLM(
        model_name=config["model"]["name"],
        load_in_4bit=config["model"]["load_in_4bit"]
    )

    retriever = Retriever(
        embedder,
        vector_store,
        top_k=config["retrieval"]["top_k"]
    )

    prompt_builder = PromptBuilder()

    print("\n✅ RAG system ready.\n")

    while True:
        query = input("Ask a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        retrieved_chunks = retriever.retrieve(query)

        prompt = prompt_builder.build(query, retrieved_chunks)

        tokenized = llm.tokenizer(prompt, return_tensors="pt")
        print(f"Prompt tokens: {tokenized['input_ids'].shape[1]}")

        start = time.time()

        answer = llm.generate(
            prompt,
            max_new_tokens=config["generation"]["max_new_tokens"],
            repetition_penalty=config["generation"]["repetition_penalty"]
        )

        end = time.time()

        print(f"\n⏱ Generation time: {end - start:.2f} seconds")

        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[-1].strip()

        print("\n=== Answer ===\n")
        print(answer)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()