import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.components.embedding_factory import create_embeddings
from src.components.vectorstore_factory import save_vectorstore
from src.core.config import load_config


def load_documents(folder):

    docs = []

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, file))
            docs.extend(loader.load())

    return docs


def main():

    config = load_config()

    raw_path = "data/raw"

    docs = load_documents(raw_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(docs)

    embeddings = create_embeddings(config["embedding"]["model_name"])

    save_vectorstore(
        split_docs,
        embeddings,
        config["vectorstore"]["path"]
    )

    print(f"Indexed {len(split_docs)} chunks.")


if __name__ == "__main__":
    main()