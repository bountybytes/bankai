import fitz  # PyMuPDF — fastest PDF library (avg 0.1s/page)
from pathlib import Path

def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")  # plain text, fastest mode
        pages.append(text)
    doc.close()
    return "\n".join(pages)

def extract_tables(pdf_path: str) -> list[dict]:
    """Extract tables if present (for structured bank statements)"""
    doc = fitz.open(pdf_path)
    all_tables = []
    for page in doc:
        tabs = page.find_tables()
        for tab in tabs:
            all_tables.extend(tab.extract())
    doc.close()
    return all_tables