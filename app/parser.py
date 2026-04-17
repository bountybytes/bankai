"""
parser.py — Hybrid AI parser
  Account header  → Qwen2.5-Coder-14B GGUF  (plain text → JSON)
  Transactions    → GLM-OCR (page images → OCR text) + Qwen (OCR text → JSON)
"""

import os, re, json, logging, time, tempfile
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
GLM_MAX_NEW      = int(os.getenv("GLM_MAX_NEW_TOKENS", "8192"))
PAGE_DPI         = int(os.getenv("PAGE_DPI",          "200"))

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
    # ✅ Correct class: GlmOcrForConditionalGeneration (native in transformers, no trust_remote_code needed)
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration
    log.info(f"Loading GLM-OCR: {GLM_OCR_PATH}")
    _glm_processor = AutoProcessor.from_pretrained(GLM_OCR_PATH)
    _glm_model = GlmOcrForConditionalGeneration.from_pretrained(
        GLM_OCR_PATH,
        dtype      = torch.bfloat16,
        device_map = "auto",
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
- balances          : plain number, no commas, no ₹  e.g. "21675.91"
- currency          : "INR"
- bank_type         : "Public" / "Private" / "Co-operative"

TEXT:
"""

TXN_FROM_OCR_PROMPT = """\
You are a precise JSON extractor for Indian bank transactions.
Below is raw OCR text from one page of a bank statement.
Extract EVERY transaction row. Output ONLY a valid JSON array. No explanation, no markdown.

CRITICAL — debit/credit rules:
- Each transaction has EITHER debit OR credit — NEVER both filled
- "WDL TFR", "UPI/DR" → DEBIT (money out)
- "DEP TFR", "UPI/CR", "UPI/REV" → CREDIT (money in)
- The third amount column is always "balance" — do NOT copy it into credit or debit
- transaction_type must be "DEBIT" when debit has value, "CREDIT" when credit has value

Schema per object:
{"date":"","value_date":"","description":"","narration":"","cheque_no":"",
 "reference_no":"","transaction_type":"","debit":"","credit":"","balance":"",
 "branch_code":"","remarks":""}

Rules:
- date/value_date : YYYY-MM-DD (convert DD/MM/YYYY or DD-MM-YYYY)
- amounts         : strip commas and ₹, plain number e.g. "1250.00"
- description     : single line, no newlines
- Skip rows that are only column headers (Date, Narration, Debit, Credit, Balance)
- If no transactions visible output: []

OCR TEXT:
"""

# ── GLM-OCR: page image → OCR text ────────────────────────────────────────────
def _render_page(pdf_path: str, page_num: int):
    """Render a PDF page to a PIL Image."""
    import fitz
    from PIL import Image
    doc  = fitz.open(pdf_path)
    page = doc[page_num]
    mat  = fitz.Matrix(PAGE_DPI / 72, PAGE_DPI / 72)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    doc.close()
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def _glm_ocr_page(pdf_path: str, page_num: int) -> str:
    """
    Run GLM-OCR on one PDF page image → returns raw OCR text.
    Uses the official GlmOcrForConditionalGeneration API pattern.
    """
    import torch
    _load_glm()

    img = _render_page(pdf_path, page_num)

    # Save page image to temp file — GLM-OCR expects a file URL in the message
    tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp_img.name)
    tmp_img.close()

    # ✅ Official GLM-OCR message format from HuggingFace docs
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": tmp_img.name},
                {"type": "text",  "text": "Text Recognition:"},
            ],
        }
    ]

    try:
        # ✅ Official pattern: apply_chat_template → generate → decode
        inputs = _glm_processor.apply_chat_template(
            messages,
            tokenize              = True,
            add_generation_prompt = True,
            return_dict           = True,
            return_tensors        = "pt",
        ).to(_glm_model.device)
        inputs.pop("token_type_ids", None)  # ✅ must remove — causes errors if present

        t0 = time.time()
        with torch.no_grad():
            generated_ids = _glm_model.generate(
                **inputs,
                max_new_tokens = GLM_MAX_NEW,
            )
        elapsed = time.time() - t0

        input_len  = inputs["input_ids"].shape[1]
        # ✅ skip_special_tokens=False per official GLM-OCR example
        ocr_text = _glm_processor.decode(
            generated_ids[0][input_len:],
            skip_special_tokens = False,
        ).strip()

        out_tokens = generated_ids.shape[1] - input_len
        log.info(f"[glm-page-{page_num+1}] OCR: {out_tokens} tokens in {elapsed:.1f}s  ({len(ocr_text)} chars)")
        log.debug(f"[glm-page-{page_num+1}] OCR preview:\n{ocr_text[:500]}")
        return ocr_text

    finally:
        os.unlink(tmp_img.name)

# ── Qwen: OCR text → structured JSON ──────────────────────────────────────────
def _qwen_parse_ocr(ocr_text: str, page_label: str) -> list:
    """Convert GLM-OCR output text → structured JSON transactions via Qwen."""
    _load_qwen()

    # Strip any remaining GLM special tokens before sending to Qwen
    clean = re.sub(r"<\|[^|]*\|>", "", ocr_text).strip()
    if not clean:
        log.warning(f"[{page_label}] Empty OCR after cleaning — skipping")
        return []

    t0   = time.time()
    resp = _qwen.create_chat_completion(
        messages       = [{"role": "user", "content": TXN_FROM_OCR_PROMPT + clean}],
        max_tokens     = 4096,
        temperature    = 0.0,
        repeat_penalty = 1.05,
    )
    raw    = resp["choices"][0]["message"]["content"].strip()
    finish = resp["choices"][0].get("finish_reason", "?")
    log.info(f"[{page_label}] Qwen parse: {time.time()-t0:.1f}s  finish={finish}")
    if finish == "length":
        log.warning(f"[{page_label}] ⚠️ Qwen output cut off")

    return _extract_json_array(raw, page_label)

# ── Public API ─────────────────────────────────────────────────────────────────
def parse_transactions(text: str, pdf_path: str = None) -> list:
    if pdf_path:
        return _parse_with_glm_plus_qwen(pdf_path)
    log.warning("pdf_path not provided — Qwen text-only fallback")
    return _parse_with_qwen_text(text)

def _parse_with_glm_plus_qwen(pdf_path: str) -> list:
    import fitz
    doc     = fitz.open(pdf_path)
    n_pages = len(doc)
    doc.close()
    log.info(f"[pipeline] GLM-OCR + Qwen — {n_pages} pages")

    all_txns = []
    for page_num in range(n_pages):
        label    = f"page-{page_num+1}/{n_pages}"
        ocr_text = _glm_ocr_page(pdf_path, page_num)       # Vision → text
        rows     = _qwen_parse_ocr(ocr_text, label)        # Text → JSON
        rows     = _post_process(rows)
        before   = len(all_txns)
        all_txns = _dedupe(all_txns + rows)
        log.info(f"[{label}] +{len(all_txns)-before} new rows  total={len(all_txns)}")

    log.info(f"[pipeline] DONE — {len(all_txns)} transactions")
    return all_txns

# ── Qwen header ────────────────────────────────────────────────────────────────
def parse_header(header_text: str) -> dict:
    _load_qwen()
    t0   = time.time()
    resp = _qwen.create_chat_completion(
        messages       = [{"role": "user", "content": HEADER_PROMPT + header_text[:MAX_HEADER_CHARS]}],
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

# ── Qwen text-only fallback ────────────────────────────────────────────────────
_CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "6000"))
_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))

def _parse_with_qwen_text(text: str) -> list:
    _load_qwen()
    chunks, all_txns = _chunk_text(text, _CHUNK_SIZE, _CHUNK_OVERLAP), []
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
        all_txns = _dedupe(all_txns + _post_process(_extract_json_array(raw, label)))
    return all_txns

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
            v = str(row.get(f) or "").replace(",","").replace("₹","").replace("-","").strip()
            row[f] = v if _valid_amount(v) else ""

        debit, credit = row.get("debit","").strip(), row.get("credit","").strip()
        txn_type      = row.get("transaction_type","").upper()
        desc          = row.get("description","").upper()

        if debit and credit:
            is_cr = txn_type == "CREDIT" or desc.startswith("DEP") or "UPI/CR" in desc or "UPI/REV" in desc
            row["debit"], row["credit"]      = ("", debit) if is_cr else (debit, "")
            row["transaction_type"]          = "CREDIT" if is_cr else "DEBIT"
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

def _valid_amount(v: str) -> bool:
    try:
        return float(v) > 0
    except (ValueError, TypeError):
        return False

def _normalize_date(s: str) -> str:
    s = s.strip()
    if not s or re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    for fmt in ("%d/%m/%Y","%d-%m-%Y","%d/%m/%y","%d-%m-%y",
                "%d %b %Y","%d-%b-%Y","%d/%b/%Y","%d %B %Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    log.warning(f"Could not normalize date: {s}")
    return s

# ── JSON helpers ───────────────────────────────────────────────────────────────
def _strip_fences(raw: str) -> str:
    return re.sub(r"```(?:json)?\s*", "", raw).strip()

def _extract_json_object(raw: str, label: str) -> dict:
    raw = _strip_fences(raw)
    s, e = raw.find("{"), raw.rfind("}")
    if s == -1 or e <= s:
        log.error(f"[{label}] No JSON object\n{raw[:400]}")
        return {}
    try:
        return json.loads(raw[s:e+1])
    except json.JSONDecodeError as err:
        log.error(f"[{label}] JSON error: {err}")
        return {}

def _extract_json_array(raw: str, label: str) -> list:
    raw = _strip_fences(raw)
    s, e = raw.find("["), raw.rfind("]")
    if s == -1 or e <= s:
        log.warning(f"[{label}] No JSON array — 0 rows\nRaw: {raw[:300]}")
        return []
    try:
        result = json.loads(raw[s:e+1])
        return result if isinstance(result, list) else []
    except json.JSONDecodeError as err:
        log.error(f"[{label}] JSON decode error: {err}\n{raw[s:s+400]}")
        return []

def _dedupe(txns: list) -> list:
    seen, out = set(), []
    for t in txns:
        k = (t.get("date",""), t.get("description","")[:50], t.get("debit",""), t.get("credit",""), t.get("balance",""))
        if k not in seen:
            seen.add(k)
            out.append(t)
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
