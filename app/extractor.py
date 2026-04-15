import fitz   # PyMuPDF
import re
from pathlib import Path


# ── Sensitive-field anonymisation ─────────────────────────────────────────────
def anonymize_sensitive(text: str) -> tuple[str, dict]:
    """
    Redact PAN numbers and mobile numbers before they reach the LLM.
    Returns (anonymised_text, restore_mapping).
    """
    mapping: dict = {}

    # PAN  — AAAAA9999A
    for i, pan in enumerate(set(re.findall(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', text))):
        ph = f"PAN_REDACTED_{i}"
        mapping[ph] = pan
        text = text.replace(pan, ph)

    # 10-digit mobile numbers
    for i, mob in enumerate(set(re.findall(r'\b[6-9]\d{9}\b', text))):
        ph = f"MOBILE_REDACTED_{i}"
        mapping[ph] = mob
        text = text.replace(mob, ph)

    return text, mapping


def restore_sensitive(data: dict, mapping: dict) -> dict:
    """Restore real PAN / mobile values back into extracted JSON after LLM."""
    import json
    s = json.dumps(data, ensure_ascii=False)
    for placeholder, real_value in mapping.items():
        s = s.replace(placeholder, real_value)
    return json.loads(s)


# ── PDF text extraction ────────────────────────────────────────────────────────
def extract_text(pdf_path: str) -> str:
    """
    Extract all text from every page of the PDF.
    Uses 'text' mode (plain, fastest — ~0.1s per page).
    """
    doc   = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages)


def extract_tables(pdf_path: str) -> list[dict]:
    """
    Extract tables from the PDF using PyMuPDF's table finder.
    Useful for structured/columnar bank statements.
    """
    doc        = fitz.open(pdf_path)
    all_tables = []
    for page in doc:
        tabs = page.find_tables()
        for tab in tabs:
            all_tables.extend(tab.extract())
    doc.close()
    return all_tables