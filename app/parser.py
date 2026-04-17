"""
parser.py — Hybrid AI parser
  Account header  → Qwen2.5-Coder-14B GGUF  (plain text from page 1-2 → JSON)
  Transactions    → GLM-OCR                  (page images → JSON per page)

Fixes:
  - Account details: now uses plain page-1 text, not pipe-delimited table output
  - Debit/credit mutual exclusivity enforced in post-processing
  - Date normalization to YYYY-MM-DD in post-processing
  - Newlines stripped from description/narration fields
"""

import os, re, json, logging, time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("bank_ai.parser")

# ── Env config ─────────────────────────────────────────────────────────────────
MODEL_PATH       = os.getenv("MODEL_PATH",       "/workspace/models/qwen2.5-coder-14b-instruct-q4_k_m.gguf")
GLM_OCR_PATH     = os.getenv("GLM_OCR_PATH",     "/workspace/models/GLM-OCR")
N_GPU_LAYERS     = int(os.getenv("N_GPU_LAYERS",     "-1"))
N_CTX            = int(os.getenv("N_CTX",            "65536"))
MAX_HEADER_CHARS = int(os.getenv("MAX_HEADER_CHARS", "8000"))
GLM_MAX_NEW      = int(os.getenv("GLM_MAX_NEW_TOKENS", "2048"))
GLM_DEVICE       = os.getenv("GLM_DEVICE", "cuda")
PAGE_DPI         = int(os.getenv("PAGE_DPI", "150"))

_qwen          = None
_glm_model     = None
_glm_processor = None

# ── Model loaders ──────────────────────────────────────────────────────────────
def _load_qwen():
    global _qwen
    if _qwen is not None:
        return
    from llama_cpp import Llama
    log.info(f"Loading Qwen GGUF: {MODEL_PATH}")
    _qwen = Llama(
        model_path   = MODEL_PATH,
        n_ctx        = N_CTX,
        n_gpu_layers = N_GPU_LAYERS,
        n_threads    = 8,
        flash_attn   = True,
        verbose      = False,
    )
    log.info("Qwen2.5-Coder-14B ready ✓")

def _load_glm():
    global _glm_model, _glm_processor
    if _glm_model is not None:
        return
    from transformers import AutoProcessor, AutoModelForImageTextToText
    log.info(f"Loading GLM-OCR: {GLM_OCR_PATH}")
    _glm_processor = AutoProcessor.from_pretrained(GLM_OCR_PATH, trust_remote_code=True)
    _glm_model = AutoModelForImageTextToText.from_pretrained(
        GLM_OCR_PATH,
        dtype      = "auto",
        device_map = "auto",
        trust_remote_code = True,
    )
    _glm_model.eval()
    log.info("GLM-OCR ready ✓")

def load_model():
    _load_qwen()
    _load_glm()

# ── Prompts ────────────────────────────────────────────────────────────────────
HEADER_PROMPT = """\
You are a precise JSON extractor for Indian bank statements.
Extract the account and statement header details from the text below.
Output ONLY a valid JSON object. No explanation. No markdown. No extra text.

Exact schema (use "" for missing fields):
{
  "bank_name": "",
  "branch_name": "",
  "bank_type": "",
  "account_number": "",
  "account_holder": "",
  "address": "",
  "pan": "",
  "mobile": "",
  "email": "",
  "ifsc_code": "",
  "micr_code": "",
  "account_type": "",
  "statement_from": "",
  "statement_to": "",
  "opening_balance": "",
  "closing_balance": "",
  "currency": ""
}

Rules:
- statement_from / statement_to : YYYY-MM-DD format only
- opening_balance / closing_balance : plain number, no commas, no ₹  e.g. "21675.91"
- currency : always "INR" for Indian bank statements
- account_type : "Savings", "Current", "OD" etc
- bank_type : "Public", "Private", "Co-operative" etc
- Look for: account number, customer name, branch, IFSC, MICR, address, PAN, mobile, email
- Look for: statement period (from/to dates), opening balance, closing balance

TEXT:
"""

GLM_TXN_PROMPT = """\
Extract ALL bank transaction rows from this bank statement page image.
Output ONLY a valid JSON array. No markdown, no explanation.
If no transactions are visible on this page output exactly: []

CRITICAL RULES FOR DEBIT/CREDIT:
- Each transaction has EITHER a debit amount OR a credit amount — NEVER both
- "debit" = money going OUT of account (withdrawals, payments, WDL TFR, UPI/DR)
- "credit" = money coming IN to account (deposits, receipts, DEP TFR, UPI/CR)
- The THIRD amount column is always "balance" — do NOT put it in credit or debit
- If you see: DATE | DESCRIPTION | 750.00 | | 1268.80 → debit="750.00" credit="" balance="1268.80"
- If you see: DATE | DESCRIPTION | | 750.00 | 1268.80 → debit="" credit="750.00" balance="1268.80"
- transaction_type must be "DEBIT" when debit has value, "CREDIT" when credit has value

Each object schema:
{
  "date": "",
  "value_date": "",
  "description": "",
  "narration": "",
  "cheque_no": "",
  "reference_no": "",
  "transaction_type": "",
  "debit": "",
  "credit": "",
  "balance": "",
  "branch_code": "",
  "remarks": ""
}

More rules:
- date / value_date   : YYYY-MM-DD (convert DD/MM/YYYY, DD-MM-YYYY, DD-MMM-YYYY)
- All amounts         : strip commas and ₹ symbol, plain number e.g. "1250.00"
- description         : single line, join any wrapped text with space
- Skip rows where description is only "Date", "Narration", "Debit", "Credit", "Balance"
- Use "" for absent fields
"""

# ── Qwen: account header ───────────────────────────────────────────────────────
def parse_header(header_text: str) -> dict:
    """
    Parse account details from plain text (page 1-2 of PDF).
    Pass header_text from extractor.extract_header_text(), NOT the full pipe-delimited text.
    """
    _load_qwen()
    full = HEADER_PROMPT + header_text[:MAX_HEADER_CHARS]
    t0 = time.time()
    resp = _qwen.create_chat_completion(
        messages       = [{"role": "user", "content": full}],
        max_tokens     = 1024,
        temperature    = 0.0,
        repeat_penalty = 1.05,
    )
    raw    = resp["choices"][0]["message"]["content"].strip()
    finish = resp["choices"][0].get("finish_reason", "?")
    log.info(f"[header] done in {time.time()-t0:.1f}s  finish={finish}")
    log.debug(f"[header] raw output:\n{raw[:500]}")

    result = _extract_json_object(raw, "header")
    for f in ("opening_balance", "closing_balance"):
        if result.get(f):
            result[f] = str(result[f]).replace(",", "").replace("₹", "").strip()
    # Normalize dates
    for f in ("statement_from", "statement_to"):
        if result.get(f):
            result[f] = _normalize_date(result[f])
    return result

# ── GLM-OCR: per-page transaction extraction ───────────────────────────────────
def _render_page(pdf_path: str, page_num: int):
    import fitz
    from PIL import Image
    doc  = fitz.open(pdf_path)
    page = doc[page_num]
    mat  = fitz.Matrix(PAGE_DPI / 72, PAGE_DPI / 72)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    doc.close()
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def _glm_extract_page(pdf_path: str, page_num: int) -> list:
    import torch, tempfile, os
    _load_glm()

    img = _render_page(pdf_path, page_num)

    # Save to temp file for GLM-OCR
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
        img.save(tmp_img.name)
        tmp_img_path = tmp_img.name

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": tmp_img_path},
                {"type": "text",  "text": GLM_TXN_PROMPT},
            ],
        }
    ]

    try:
        inputs = _glm_processor.apply_chat_template(
            messages,
            tokenize              = True,
            add_generation_prompt = True,
            return_dict           = True,
            return_tensors        = "pt",
        ).to(_glm_model.device)
        inputs.pop("token_type_ids", None)

        t0 = time.time()
        with torch.no_grad():
            out_ids = _glm_model.generate(
                **inputs,
                max_new_tokens = GLM_MAX_NEW,
                do_sample      = False,
            )
        elapsed   = time.time() - t0
        input_len = inputs["input_ids"].shape[1]
        raw = _glm_processor.decode(
            out_ids[0][input_len:], skip_special_tokens=False
        ).strip()

        out_tokens = out_ids.shape[1] - input_len
        log.info(f"[glm-page-{page_num+1}] {out_tokens} tokens in {elapsed:.1f}s")
        return _extract_json_array(raw, f"glm-page-{page_num+1}")
    finally:
        os.unlink(tmp_img_path)

# ── Public: parse_transactions ─────────────────────────────────────────────────
def parse_transactions(text: str, pdf_path: str = None) -> list:
    if pdf_path:
        return _parse_with_glm(pdf_path)
    log.warning("pdf_path not provided — using Qwen chunked fallback")
    return _parse_with_qwen_fallback(text)

def _parse_with_glm(pdf_path: str) -> list:
    import fitz
    doc     = fitz.open(pdf_path)
    n_pages = len(doc)
    doc.close()
    log.info(f"[glm-txn] Processing {n_pages} pages")

    all_txns: list = []
    for page_num in range(n_pages):
        rows   = _glm_extract_page(pdf_path, page_num)
        rows   = _post_process_transactions(rows)
        before = len(all_txns)
        all_txns = _dedupe(all_txns + rows)
        log.info(f"[glm-page-{page_num+1}/{n_pages}] +{len(all_txns)-before} rows  total={len(all_txns)}")

    log.info(f"[glm-txn] DONE — {len(all_txns)} transactions")
    return all_txns

# ── Post-processing — fixes all 4 known GLM-OCR issues ────────────────────────
def _post_process_transactions(rows: list) -> list:
    cleaned = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        # 1. Clean newlines from text fields
        for f in ("description", "narration", "remarks"):
            if row.get(f):
                row[f] = " ".join(str(row[f]).split())

        # 2. Clean amounts
        for f in ("debit", "credit", "balance"):
            if row.get(f):
                val = str(row[f]).replace(",", "").replace("₹", "").replace("-", "").strip()
                row[f] = val if _is_valid_amount(val) else ""

        # 3. Enforce debit/credit mutual exclusivity
        #    GLM often puts balance value into credit column for debit rows
        debit  = row.get("debit", "").strip()
        credit = row.get("credit", "").strip()
        txn_type = row.get("transaction_type", "").upper().strip()

        if debit and credit:
            # Both filled — use transaction_type or description prefix to decide
            desc = row.get("description", "").upper()
            is_credit_txn = (
                txn_type == "CREDIT"
                or desc.startswith("DEP")
                or "UPI/CR" in desc
                or "UPI/REV" in desc
                or "CREDIT" in desc
            )
            if is_credit_txn:
                row["debit"]            = ""
                row["transaction_type"] = "CREDIT"
            else:
                row["credit"]           = ""
                row["transaction_type"] = "DEBIT"

        # 4. Set transaction_type if missing
        if not row.get("transaction_type"):
            if row.get("debit"):
                row["transaction_type"] = "DEBIT"
            elif row.get("credit"):
                row["transaction_type"] = "CREDIT"

        # 5. Normalize dates to YYYY-MM-DD
        for f in ("date", "value_date"):
            if row.get(f):
                row[f] = _normalize_date(str(row[f]))

        # 6. Skip rows with no amount at all (header rows etc.)
        if not row.get("debit") and not row.get("credit"):
            continue

        cleaned.append(row)
    return cleaned

def _is_valid_amount(val: str) -> bool:
    try:
        float(val)
        return float(val) > 0
    except (ValueError, TypeError):
        return False

def _normalize_date(date_str: str) -> str:
    """Normalize any common Indian date format to YYYY-MM-DD."""
    date_str = date_str.strip()
    if not date_str:
        return ""
    # Already correct
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return date_str
    formats = [
        "%d/%m/%Y", "%d-%m-%Y",
        "%d/%m/%y", "%d-%m-%y",
        "%d %b %Y", "%d-%b-%Y", "%d/%b/%Y",
        "%d %B %Y", "%B %d, %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    log.warning(f"Could not normalize date: {date_str}")
    return date_str

# ── Qwen chunked fallback (no pdf_path) ───────────────────────────────────────
_QWEN_TXN_PROMPT = """\
You are a precise JSON extractor for Indian bank transactions.
Input: pipe-delimited table rows from a bank statement PDF.
Output ONLY a valid JSON array. No explanation, no markdown.

Schema per object:
{"date":"","value_date":"","description":"","narration":"","cheque_no":"",
 "reference_no":"","transaction_type":"","debit":"","credit":"","balance":"",
 "branch_code":"","remarks":""}

Rules:
- date/value_date   : YYYY-MM-DD
- transaction_type  : "DEBIT" or "CREDIT" — never both filled
- debit/credit      : mutually exclusive — only one has a value per row
- amounts           : strip commas and ₹, plain number
- Skip header rows
- Include EVERY row

BANK STATEMENT TABLE:
"""
_CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "6000"))
_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))

def _parse_with_qwen_fallback(text: str) -> list:
    _load_qwen()
    chunks   = _chunk_text(text, _CHUNK_SIZE, _CHUNK_OVERLAP)
    all_txns: list = []
    for i, chunk in enumerate(chunks):
        label = f"qwen-chunk-{i+1}/{len(chunks)}"
        resp  = _qwen.create_chat_completion(
            messages       = [{"role": "user", "content": _QWEN_TXN_PROMPT + chunk}],
            max_tokens     = 4096,
            temperature    = 0.0,
            repeat_penalty = 1.05,
        )
        raw    = resp["choices"][0]["message"]["content"].strip()
        finish = resp["choices"][0].get("finish_reason", "?")
        if finish == "length":
            log.warning(f"[{label}] ⚠️ output cut off")
        txns     = _extract_json_array(raw, label)
        txns     = _post_process_transactions(txns)
        all_txns = _dedupe(all_txns + txns)
    return all_txns

# ── JSON helpers ───────────────────────────────────────────────────────────────
def _strip_fences(raw: str) -> str:
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*",     "", raw)
    return raw.strip()

def _extract_json_object(raw: str, label: str) -> dict:
    raw   = _strip_fences(raw)
    start = raw.find("{")
    end   = raw.rfind("}")
    if start == -1 or end <= start:
        log.error(f"[{label}] No JSON object\n{raw[:400]}")
        return {}
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError as e:
        log.error(f"[{label}] JSON error: {e}")
        return {}

def _extract_json_array(raw: str, label: str) -> list:
    raw   = _strip_fences(raw)
    start = raw.find("[")
    end   = raw.rfind("]")
    if start == -1 or end <= start:
        log.warning(f"[{label}] No JSON array — 0 rows")
        return []
    try:
        result = json.loads(raw[start:end + 1])
        return result if isinstance(result, list) else []
    except json.JSONDecodeError as e:
        log.error(f"[{label}] JSON error: {e}\n{raw[start:start+300]}")
        return []

# ── Dedup + utils ──────────────────────────────────────────────────────────────
def _dedupe(transactions: list) -> list:
    seen, out = set(), []
    for txn in transactions:
        key = (
            txn.get("date", ""),
            txn.get("description", "")[:50],
            txn.get("debit", ""),
            txn.get("credit", ""),
            txn.get("balance", ""),
        )
        if key not in seen:
            seen.add(key)
            out.append(txn)
    return out

def _chunk_text(text: str, size: int, overlap: int) -> list:
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        if end < len(text):
            nl = text.rfind("\n", start + size - overlap, end)
            if nl != -1:
                end = nl
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else len(text)
    return chunks
