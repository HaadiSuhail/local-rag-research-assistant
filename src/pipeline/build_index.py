from pathlib import Path
from typing import List

from src.ingestion.pdf_loader import extract_text_from_pdf
from src.chunking.chunker import chunk_text
from src.embeddings.hf_embedder import HFEmbedder
from src.vector_store.faiss_store import FaissVectorStore
from src.utils.config_loader import load_config

config = load_config()

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("data/index")


def extract_and_save_pdfs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for pdf_path in RAW_DIR.glob("*.pdf"):
        print(f"📄 Processing PDF: {pdf_path.name}")

        text = extract_text_from_pdf(pdf_path)

        output_path = PROCESSED_DIR / f"{pdf_path.stem}.txt"
        output_path.write_text(text, encoding="utf-8")

        print(f"✔ Saved processed text: {output_path.name}")


def load_and_chunk_texts() -> List[dict]:
    all_chunks = []

    for text_file in PROCESSED_DIR.glob("*.txt"):
        print(f"✂ Chunking: {text_file.name}")

        text = text_file.read_text(encoding="utf-8")
        
        chunks = chunk_text(
            text,
            chunk_size=config["chunking"]["chunk_size"],
            overlap=config["chunking"]["overlap"]
        )

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "source": text_file.stem,
                "chunk_id": idx,
                "text": chunk.strip()
            })

    print(f"🔹 Total chunks created: {len(all_chunks)}")
    return all_chunks


def build_index():
    print("🚀 Starting indexing pipeline")

    # Step 1: Extract PDFs
    extract_and_save_pdfs()

    # Step 2: Chunk texts
    chunks = load_and_chunk_texts()

    texts = [c["text"] for c in chunks]

    # Step 3: Embed
    embedder = HFEmbedder()
    embeddings = embedder.encode(texts, batch_size=16)

    # Step 4: Build FAISS index
    vector_store = FaissVectorStore(embedding_dim=embeddings.shape[1])
    vector_store.add(embeddings, chunks)

    # Step 5: Save index
    vector_store.save(INDEX_DIR)

    print("✅ Index built and saved successfully.")


if __name__ == "__main__":
    build_index()