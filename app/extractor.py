"""
extractor.py — PDF text + table extraction with structured output
Priority: table extraction → structured text → fallback plain text
"""

import fitz
import re
import json
from pathlib import Path


# ── Anonymisation ──────────────────────────────────────────────────────────────
def anonymize_sensitive(text: str) -> tuple[str, dict]:
    mapping: dict = {}
    for i, pan in enumerate(set(re.findall(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', text))):
        ph = f"PAN_REDACTED_{i}"
        mapping[ph] = pan
        text = text.replace(pan, ph)
    for i, mob in enumerate(set(re.findall(r'\b[6-9]\d{9}\b', text))):
        ph = f"MOBILE_REDACTED_{i}"
        mapping[ph] = mob
        text = text.replace(mob, ph)
    return text, mapping


def restore_sensitive(data, mapping: dict):
    s = json.dumps(data, ensure_ascii=False)
    for placeholder, real_value in mapping.items():
        s = s.replace(placeholder, real_value)
    return json.loads(s)


# ── Table → structured text ────────────────────────────────────────────────────
def _table_to_text(rows: list) -> str:
    """Convert a 2D table (list of lists) into pipe-delimited text rows."""
    lines = []
    for row in rows:
        cleaned = [str(cell).strip() if cell is not None else "" for cell in row]
        if any(c for c in cleaned):  # skip fully empty rows
            lines.append(" | ".join(cleaned))
    return "\n".join(lines)


def _is_header_row(row: list) -> bool:
    """Detect if a row looks like a column header."""
    keywords = {"date", "narration", "description", "debit", "credit",
                "balance", "ref", "chq", "withdrawal", "deposit", "particulars",
                "value", "txn", "transaction", "amount"}
    text = " ".join(str(c or "").lower() for c in row)
    return any(kw in text for kw in keywords)


# ── Main extraction ────────────────────────────────────────────────────────────
def extract_text(pdf_path: str) -> str:
    """
    Extract text from PDF with table-aware strategy:
    1. Try PyMuPDF table finder on each page — gives clean columnar data
    2. Fall back to plain text for pages with no tables
    3. Combine: header/footer as plain text + tables as structured pipe-delimited rows
    """
    doc   = fitz.open(pdf_path)
    parts = []

    for page_num, page in enumerate(doc):
        page_text  = page.get_text("text").strip()
        table_text = ""

        try:
            tabs = page.find_tables()
            table_parts = []
            for tab in tabs:
                rows = tab.extract()
                if not rows:
                    continue
                # Keep header row, convert rest to pipe-delimited
                t = _table_to_text(rows)
                if t.strip():
                    table_parts.append(t)
            if table_parts:
                table_text = "\n\n".join(table_parts)
        except Exception:
            pass  # Table extraction failed — fall back to plain text

        if table_text:
            # Use structured table output — much better for LLM parsing
            parts.append(f"--- PAGE {page_num + 1} ---\n{table_text}")
        else:
            # No tables found — use plain text
            if page_text:
                parts.append(f"--- PAGE {page_num + 1} ---\n{page_text}")

    doc.close()
    return "\n\n".join(parts)


def extract_text_plain(pdf_path: str) -> str:
    """Plain text extraction without table detection (fallback)."""
    doc   = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return "\n".join(pages)

def extract_header_text(pdf_path: str) -> str:
    """
    Extract plain text from the FIRST 2 pages only — used for account header parsing.
    SBI and most Indian banks put account details on page 1 before the transaction table.
    Returns plain text (not pipe-delimited) so Qwen can read it naturally.
    """
    doc = fitz.open(pdf_path)
    pages_text = []
    for i, page in enumerate(doc):
        if i >= 2:   # only first 2 pages
            break
        pages_text.append(page.get_text("text").strip())
    doc.close()
    return "\n\n".join(pages_text)
