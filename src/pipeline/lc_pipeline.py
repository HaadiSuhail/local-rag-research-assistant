from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

from src.utils.config_loader import load_config


INDEX_DIR = Path("data/lc_index")


def load_llm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16
    )

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=120,
        do_sample=False,
        return_full_text=False,
    )

    return HuggingFacePipeline(pipeline=text_gen_pipeline)


def main():
    config = load_config()

    print("🔹 Loading FAISS index (LangChain wrapper)...")
    print("🔹 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )

    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": config["retrieval"]["top_k"]}
    )

    print("🔹 Loading LLM...")
    llm = load_llm(config["model"]["name"])

    prompt_template = """
You are a research assistant.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    print("\n✅ LangChain RAG ready.\n")

    while True:
        query = input("Ask (or type exit): ")

        if query.lower() == "exit":
            break

        result = qa_chain.invoke({"query": query})

        print("\n=== Answer ===\n")
        print(result["result"])
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()