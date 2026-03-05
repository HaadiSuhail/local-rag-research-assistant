from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.components.embedding_factory import create_embeddings
from src.components.llm_factory import create_llm
from src.components.vectorstore_factory import load_vectorstore
from src.components.reranker import Reranker

import time
import os


class RAGSystem:

    def __init__(self, config):

        self.config = config

        self.reranker = Reranker()

        # Embeddings
        self.embeddings = create_embeddings(config["embedding"]["model_name"])

        # Vectorstore
        self.vectorstore = load_vectorstore(
            config["vectorstore"]["path"],
            self.embeddings
        )

        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": config["retrieval"]["top_k"],
                "fetch_k": 20
            }
        )

        # LLM
        self.llm = create_llm(
            config["model"]["name"],
            config["generation"]["max_new_tokens"]
        )

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=config["prompt_template"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt}
        )

    def ask(self, query: str):

        debug = self.config.get("debug", {}).get("rag_tracing", False)

        if debug:
            print("\n===== RAG TRACE START =====")
            print("Query:", query)

        # Retrieve docs
        docs = self.retriever.invoke(query)

        # rerank retrieved docs
        docs = self.reranker.rerank(query, docs, top_k=2)

        if debug:
            print("\nRetrieved Chunks:\n")

            for i, d in enumerate(docs):
                print(f"--- Chunk {i+1} ---")
                print(d.page_content[:400])
                print()

        # Build context
        context = "\n\n".join([d.page_content for d in docs])

        # Prompt preview
        prompt = self.config["prompt_template"].format(
            context=context,
            question=query
        )

        if debug:
            tokenizer = self.llm.pipeline.tokenizer
            token_count = len(tokenizer(prompt)["input_ids"])
            print("Prompt Tokens:", token_count)

        # Generate answer
        start = time.time()

        result = self.qa_chain.invoke({"query": query})

        end = time.time()

        if debug:
            print("Generation Time:", round(end - start, 2), "seconds")
            print("===== RAG TRACE END =====\n")

        return result["result"]

    def retrieve(self, query: str):
        return self.retriever.invoke(query)