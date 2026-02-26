from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.config_loader import load_config


DATA_DIR = Path("data/raw")
INDEX_DIR = Path("data/lc_index")


def main():
    config = load_config()

    print("🚀 LangChain Indexing Started")

    documents = []

    # 1️⃣ Load PDFs
    for pdf_path in DATA_DIR.glob("*.pdf"):
        print(f"📄 Loading {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())

    # 2️⃣ Split into chunks
    print("✂ Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["overlap"],
    )

    split_docs = splitter.split_documents(documents)

    print(f"🔹 Total chunks: {len(split_docs)}")

    # 3️⃣ Embedding model (LangChain wrapper)
    print("🔹 Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )

    # 4️⃣ Create FAISS vectorstore
    print("🔹 Building FAISS index...")
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # 5️⃣ Save in LangChain format
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))

    print("✅ LangChain FAISS index saved successfully.")


if __name__ == "__main__":
    main()