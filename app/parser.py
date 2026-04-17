"""
parser.py — Hybrid AI parser
  Account header  → Qwen2.5-Coder-14B GGUF  (plain text → JSON)
  Transactions    → GLM-OCR (page image → OCR text) + Qwen (OCR text → JSON)
"""

import os, re, json, logging, time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("bank_ai.parser")

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH       = os.getenv("MODEL_PATH",       "/workspace/models/qwen2.5-coder-14b-instruct-q4_k_m.gguf")
GLM_OCR_PATH     = os.getenv("GLM_OCR_PATH",     "/workspace/models/GLM-OCR")
N_GPU_LAYERS     = int(os.getenv("N_GPU_LAYERS",      "-1"))
N_CTX            = int(os.getenv("N_CTX",             "8192"))
MAX_HEADER_CHARS = int(os.getenv("MAX_HEADER_CHARS",  "8000"))
GLM_MAX_NEW      = int(os.getenv("GLM_MAX_NEW_TOKENS", "4096"))
PAGE_DPI         = int(os.getenv("PAGE_DPI",          "200"))  # higher DPI for GLM

_qwen          = None
_glm_model     = None
_glm_processor = None

# ── Loaders ────────────────────────────────────────────────────────────────────
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
    import torch
    from transformers import AutoProcessor, AutoModel

    log.info(f"Loading GLM-OCR: {GLM_OCR_PATH}")

    _glm_processor = AutoProcessor.from_pretrained(
        GLM_OCR_PATH,
        trust_remote_code=True,
    )

    # GLM-OCR uses AutoModel, not AutoModelForCausalLM
    # It has a custom GlmOcrConfig that only registers under AutoModel
    _glm_model = AutoModel.from_pretrained(
        GLM_OCR_PATH,
        dtype         = torch.float16,
        device_map    = "auto",
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
Extract account and statement header details from the text below.
Output ONLY a valid JSON object. No explanation. No markdown. No extra text.

Schema (use "" for missing):
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
- statement_from/to : YYYY-MM-DD
- balances : plain number, no commas, no ₹  e.g. "21675.91"
- currency : "INR"
- bank_type : "Public" / "Private" / "Co-operative"

TEXT:
"""

# Qwen prompt to convert GLM-OCR raw OCR text → JSON transactions
TXN_FROM_OCR_PROMPT = """\
You are a precise JSON extractor for Indian bank transactions.
Below is raw OCR text extracted from one page of a bank statement.
Extract EVERY transaction row. Output ONLY a valid JSON array. No explanation, no markdown.

CRITICAL: Each transaction has EITHER debit OR credit — NEVER both filled.
- "WDL TFR", "UPI/DR" = DEBIT (money out)
- "DEP TFR", "UPI/CR", "UPI/REV" = CREDIT (money in)
- Third amount column is always "balance" — do NOT put it in credit or debit

Schema per object:
{"date":"","value_date":"","description":"","narration":"","cheque_no":"",
 "reference_no":"","transaction_type":"","debit":"","credit":"","balance":"",
 "branch_code":"","remarks":""}

Rules:
- date/value_date : YYYY-MM-DD
- amounts : strip commas and ₹, plain number e.g. "1250.00"
- description : single line, no newlines
- Skip rows that are only column headers (Date, Narration, Debit, Credit, Balance)
- If no transactions visible, output: []

OCR TEXT:
"""

# ── GLM-OCR: page image → raw OCR text ────────────────────────────────────────
def _render_page(pdf_path: str, page_num: int):
    """Render a PDF page to a PIL Image at PAGE_DPI."""
    import fitz
    from PIL import Image
    doc  = fitz.open(pdf_path)
    page = doc[page_num]
    mat  = fitz.Matrix(PAGE_DPI / 72, PAGE_DPI / 72)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    doc.close()
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def _glm_ocr_page(pdf_path: str, page_num: int) -> str:
    import torch
    _load_glm()

    img = _render_page(pdf_path, page_num)

    inputs = _glm_processor(
        text   = "Transcribe ALL text in this image exactly as it appears, preserving table structure with spaces.",
        images = img,
        return_tensors = "pt",
    ).to(_glm_model.device)

    # Some processors don't set pad_token_id — use eos as fallback
    pad_id = getattr(_glm_processor.tokenizer, "pad_token_id", None) \
          or getattr(_glm_processor.tokenizer, "eos_token_id", 0)

    t0 = time.time()
    with torch.no_grad():
        out_ids = _glm_model.generate(
            **inputs,
            max_new_tokens = GLM_MAX_NEW,
            do_sample      = False,
            pad_token_id   = pad_id,
        )
    elapsed = time.time() - t0

    input_len = inputs["input_ids"].shape[1]
    ocr_text  = _glm_processor.decode(
        out_ids[0][input_len:],
        skip_special_tokens = True,
    ).strip()

    out_tokens = out_ids.shape[1] - input_len
    log.info(f"[glm-page-{page_num+1}] OCR: {out_tokens} tokens in {elapsed:.1f}s  ({len(ocr_text)} chars)")
    log.debug(f"[glm-page-{page_num+1}] OCR preview:\n{ocr_text[:400]}")
    return ocr_text

def _qwen_parse_ocr(ocr_text: str, page_label: str) -> list:
    """
    Pass GLM-OCR output text to Qwen2.5-Coder for JSON structuring.
    Qwen is excellent at this — it's a code model trained on structured extraction.
    """
    _load_qwen()
    if not ocr_text.strip():
        log.warning(f"[{page_label}] Empty OCR text — skipping Qwen")
        return []

    prompt = TXN_FROM_OCR_PROMPT + ocr_text
    t0 = time.time()
    resp = _qwen.create_chat_completion(
        messages       = [{"role": "user", "content": prompt}],
        max_tokens     = 4096,
        temperature    = 0.0,
        repeat_penalty = 1.05,
    )
    raw    = resp["choices"][0]["message"]["content"].strip()
    finish = resp["choices"][0].get("finish_reason", "?")
    elapsed = time.time() - t0
    log.info(f"[{page_label}] Qwen parse: {elapsed:.1f}s  finish={finish}")
    if finish == "length":
        log.warning(f"[{page_label}] ⚠️ Qwen output cut off — reduce page token count")

    return _extract_json_array(raw, page_label)

# ── Public: parse_transactions ─────────────────────────────────────────────────
def parse_transactions(text: str, pdf_path: str = None) -> list:
    """
    GLM-OCR extracts raw text from each page image.
    Qwen2.5-Coder converts that raw OCR text to structured JSON.
    Two-stage pipeline: GLM-OCR (vision) → Qwen (structuring).
    """
    if pdf_path:
        return _parse_with_glm_plus_qwen(pdf_path)
    log.warning("pdf_path not provided — falling back to Qwen text-only mode")
    return _parse_with_qwen_text(text)

def _parse_with_glm_plus_qwen(pdf_path: str) -> list:
    import fitz
    doc     = fitz.open(pdf_path)
    n_pages = len(doc)
    doc.close()
    log.info(f"[pipeline] GLM-OCR + Qwen — {n_pages} pages")

    all_txns: list = []
    for page_num in range(n_pages):
        page_label = f"page-{page_num+1}/{n_pages}"

        # Stage 1: GLM-OCR — image → text
        ocr_text = _glm_ocr_page(pdf_path, page_num)

        # Stage 2: Qwen — text → JSON
        rows = _qwen_parse_ocr(ocr_text, page_label)
        rows = _post_process(rows)

        before   = len(all_txns)
        all_txns = _dedupe(all_txns + rows)
        log.info(f"[{page_label}] +{len(all_txns)-before} new rows  total={len(all_txns)}")

    log.info(f"[pipeline] DONE — {len(all_txns)} transactions")
    return all_txns

# ── Qwen text-only fallback (no pdf_path) ─────────────────────────────────────
_CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "6000"))
_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))

def _parse_with_qwen_text(text: str) -> list:
    _load_qwen()
    chunks   = _chunk_text(text, _CHUNK_SIZE, _CHUNK_OVERLAP)
    all_txns = []
    for i, chunk in enumerate(chunks):
        label = f"qwen-chunk-{i+1}/{len(chunks)}"
        resp  = _qwen.create_chat_completion(
            messages       = [{"role": "user", "content": TXN_FROM_OCR_PROMPT + chunk}],
            max_tokens     = 4096,
            temperature    = 0.0,
            repeat_penalty = 1.05,
        )
        raw    = resp["choices"][0]["message"]["content"].strip()
        finish = resp["choices"][0].get("finish_reason", "?")
        if finish == "length":
            log.warning(f"[{label}] ⚠️ output cut off")
        txns     = _extract_json_array(raw, label)
        all_txns = _dedupe(all_txns + _post_process(txns))
    return all_txns

# ── Qwen: account header ───────────────────────────────────────────────────────
def parse_header(header_text: str) -> dict:
    _load_qwen()
    prompt = HEADER_PROMPT + header_text[:MAX_HEADER_CHARS]
    t0     = time.time()
    resp   = _qwen.create_chat_completion(
        messages       = [{"role": "user", "content": prompt}],
        max_tokens     = 1024,
        temperature    = 0.0,
        repeat_penalty = 1.05,
    )
    raw    = resp["choices"][0]["message"]["content"].strip()
    finish = resp["choices"][0].get("finish_reason", "?")
    log.info(f"[header] done in {time.time()-t0:.1f}s  finish={finish}")

    result = _extract_json_object(raw, "header")
    for f in ("opening_balance", "closing_balance"):
        if result.get(f):
            result[f] = str(result[f]).replace(",", "").replace("₹", "").strip()
    for f in ("statement_from", "statement_to"):
        if result.get(f):
            result[f] = _normalize_date(result[f])
    return result

# ── Post-processing ────────────────────────────────────────────────────────────
def _post_process(rows: list) -> list:
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for f in ("description", "narration", "remarks"):
            if row.get(f):
                row[f] = " ".join(str(row[f]).split())
        for f in ("debit", "credit", "balance"):
            raw_val = str(row.get(f) or "").replace(",", "").replace("₹", "").replace("-", "").strip()
            row[f]  = raw_val if _valid_amount(raw_val) else ""

        debit    = row.get("debit",  "").strip()
        credit   = row.get("credit", "").strip()
        txn_type = row.get("transaction_type", "").upper()
        desc     = row.get("description", "").upper()

        if debit and credit:
            is_cr = txn_type == "CREDIT" or desc.startswith("DEP") or "UPI/CR" in desc or "UPI/REV" in desc
            if is_cr:
                row["debit"]            = ""
                row["transaction_type"] = "CREDIT"
            else:
                row["credit"]           = ""
                row["transaction_type"] = "DEBIT"
        elif debit:
            row["transaction_type"] = "DEBIT"
        elif credit:
            row["transaction_type"] = "CREDIT"

        for f in ("date", "value_date"):
            if row.get(f):
                row[f] = _normalize_date(str(row[f]))

        if not row.get("debit") and not row.get("credit"):
            continue
        out.append(row)
    return out

def _valid_amount(val: str) -> bool:
    try:
        return float(val) > 0
    except (ValueError, TypeError):
        return False

def _normalize_date(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y",
                "%d %b %Y", "%d-%b-%Y", "%d/%b/%Y", "%d %B %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    log.warning(f"Could not normalize date: {s}")
    return s

# ── JSON helpers ───────────────────────────────────────────────────────────────
def _strip_fences(raw: str) -> str:
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*",     "", raw)
    return raw.strip()

def _extract_json_object(raw: str, label: str) -> dict:
    raw   = _strip_fences(raw)
    start = raw.find("{");  end = raw.rfind("}")
    if start == -1 or end <= start:
        log.error(f"[{label}] No JSON object found\n{raw[:400]}")
        return {}
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError as e:
        log.error(f"[{label}] JSON error: {e}")
        return {}

def _extract_json_array(raw: str, label: str) -> list:
    raw   = _strip_fences(raw)
    start = raw.find("[");  end = raw.rfind("]")
    if start == -1 or end <= start:
        log.warning(f"[{label}] No JSON array found — 0 rows")
        log.debug(f"[{label}] raw snippet: {raw[:400]}")
        return []
    try:
        result = json.loads(raw[start:end + 1])
        return result if isinstance(result, list) else []
    except json.JSONDecodeError as e:
        log.error(f"[{label}] JSON decode error: {e}\n{raw[start:start+400]}")
        return []

def _dedupe(transactions: list) -> list:
    seen, out = set(), []
    for txn in transactions:
        key = (txn.get("date",""), txn.get("description","")[:50],
               txn.get("debit",""), txn.get("credit",""), txn.get("balance",""))
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
