from pathlib import Path
from typing import List, Dict


def chunk_text(
    text: str,
    chunk_size: int = 2000,
    overlap: int = 200
) -> List[str]:
    """
    Splits text into overlapping chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def chunk_file(text_path: Path) -> List[Dict]:
    """
    Chunks a processed text file and returns structured chunks.
    """
    text = text_path.read_text(encoding="utf-8")

    raw_chunks = chunk_text(text)

    structured_chunks = []
    for idx, chunk in enumerate(raw_chunks):
        structured_chunks.append({
            "source": text_path.stem,
            "chunk_id": idx,
            "text": chunk.strip()
        })

    return structured_chunks