from langchain_community.vectorstores import FAISS
import os


def load_vectorstore(path: str, embeddings):
    return FAISS.load_local(
        folder_path=path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )


def save_vectorstore(docs, embeddings, path: str):
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(path)