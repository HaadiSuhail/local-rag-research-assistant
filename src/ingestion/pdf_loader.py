import fitz  # PyMuPDF
from pathlib import Path


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extracts text from a PDF file page by page.
    Returns a single string with page markers.
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append(f"\n\n--- Page {page_num} ---\n{text}")

    return "\n".join(pages)


def save_extracted_text(pdf_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    text = extract_text_from_pdf(pdf_path)

    output_file = output_dir / f"{pdf_path.stem}.txt"
    output_file.write_text(text, encoding="utf-8")

    print(f"✔ Extracted text saved to: {output_file}")