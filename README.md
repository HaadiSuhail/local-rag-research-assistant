# RAG v2 – Local Research Assistant (LangChain + Hugging Face)

A GPU-accelerated Retrieval-Augmented Generation (RAG) research assistant built using Hugging Face models and LangChain orchestration.

This project demonstrates a clean, modular implementation of a local-first RAG system designed for research workflows.

---

## 🚀 Features

- Local LLM inference (Phi-3 / Mistral)
- Hugging Face embeddings (BGE-large)
- FAISS vector database
- LangChain RetrievalQA pipeline
- Configurable chunking and retrieval parameters
- GPU acceleration with quantization
- Modular architecture (no SaaS dependencies)

---

## 🧠 Architecture

User Query  
→ Query Embedding (BGE)  
→ FAISS Similarity Search  
→ Context Injection  
→ Hugging Face LLM  
→ Structured Response  

LangChain handles orchestration, while embeddings and generation remain fully controlled.

---

## ⚙️ Tech Stack

- Python
- Hugging Face Transformers
- LangChain
- FAISS
- PyMuPDF
- PyTorch (CUDA)

---

## 📦 Setup

```bash
pip install -r requirements.txt

Then build index:

python -m src.pipeline.lc_build_index

Run RAG:

python -m src.pipeline.lc_pipeline