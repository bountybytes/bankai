"""
parser.py — Hybrid AI parser
  Account header  → Qwen2.5-Coder-14B GGUF  (text → JSON, ~512 output tokens, never overflows)
  Transactions    → GLM-OCR                  (page images → JSON per page, max ~2048 tokens/page)

Fix: previous single-shot Qwen call hit finish_reason=length at 16384 tokens mid-array.
GLM-OCR processes one page at a time — bounded output, JSON array always closes properly.
"""

import os, re, json, logging, time
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("bank_ai.parser")

# ── Env config ─────────────────────────────────────────────────────────────────
MODEL_PATH       = os.getenv("MODEL_PATH",       "/workspace/models/qwen2.5-coder-14b-instruct-q4_k_m.gguf")
GLM_OCR_PATH     = os.getenv("GLM_OCR_PATH",     "/workspace/models/GLM-OCR")
N_GPU_LAYERS     = int(os.getenv("N_GPU_LAYERS",     "-1"))
N_CTX            = int(os.getenv("N_CTX",            "65536"))
MAX_HEADER_CHARS = int(os.getenv("MAX_HEADER_CHARS", "6000"))
GLM_MAX_NEW      = int(os.getenv("GLM_MAX_NEW_TOKENS", "2048"))  # per page — bounded!
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
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    log.info(f"Loading GLM-OCR: {GLM_OCR_PATH}")
    _glm_processor = AutoProcessor.from_pretrained(GLM_OCR_PATH, trust_remote_code=True)
    _glm_model = AutoModelForCausalLM.from_pretrained(
        GLM_OCR_PATH,
        torch_dtype       = torch.float16,
        device_map        = GLM_DEVICE,
        trust_remote_code = True,
    )
    _glm_model.eval()
    log.info("GLM-OCR ready ✓")

def load_model():
    """Called once at startup — loads both models."""
    _load_qwen()
    _load_glm()

# ── Prompts ────────────────────────────────────────────────────────────────────
HEADER_PROMPT = """\
You are a precise JSON extractor for Indian bank statements.
Extract ONLY the account / header details from the text below.
Output ONLY a valid JSON object. No explanation. No markdown. No extra text.

Exact schema (use "" for missing fields):
{
  "bank_name": "", "branch_name": "", "bank_type": "",
  "account_number": "", "account_holder": "", "address": "",
  "pan": "", "mobile": "", "email": "",
  "ifsc_code": "", "micr_code": "", "account_type": "",
  "statement_from": "", "statement_to": "",
  "opening_balance": "", "closing_balance": "", "currency": ""
}

Rules:
- statement_from / statement_to : YYYY-MM-DD
- opening_balance / closing_balance : plain number, no commas  e.g. "21675.91"
- currency : "INR" if implied

TEXT:
"""

GLM_TXN_PROMPT = """\
You are extracting ALL bank transaction rows visible in this bank statement page image.
Output ONLY a valid JSON array — no markdown fences, no explanation, no extra text.
If there are no transactions on this page, output exactly: []

Each object must use these exact keys:
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

Rules:
- date / value_date   : YYYY-MM-DD  (convert DD/MM/YYYY, DD-MM-YYYY, DD-MMM-YYYY)
- transaction_type    : exactly "DEBIT" or "CREDIT"
- debit / credit      : mutually exclusive — one must be "" per row
- All amounts         : strip commas and ₹  e.g. "1250.00"  NOT "1,250.00"
- balance             : plain number  e.g. "1268.80"  NOT "1,268.80"
- Skip header rows    : rows containing Date / Narration / Debit / Credit / Balance
- Include EVERY data row — do NOT stop early
- Use "" for absent fields
"""

# ── Qwen: account header ───────────────────────────────────────────────────────
def parse_header(text: str) -> dict:
    _load_qwen()
    full = HEADER_PROMPT + text[:MAX_HEADER_CHARS]
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

    result = _extract_json_object(raw, "header")
    for f in ("opening_balance", "closing_balance"):
        if result.get(f):
            result[f] = str(result[f]).replace(",", "").replace("₹", "").strip()
    return result

# ── GLM-OCR: per-page transaction extraction ───────────────────────────────────
def _render_page(pdf_path: str, page_num: int):
    """Render one PDF page to a PIL Image at PAGE_DPI resolution."""
    import fitz
    from PIL import Image
    doc  = fitz.open(pdf_path)
    page = doc[page_num]
    mat  = fitz.Matrix(PAGE_DPI / 72, PAGE_DPI / 72)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    doc.close()
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def _glm_extract_page(pdf_path: str, page_num: int) -> list:
    """Run GLM-OCR on one page image → list of transaction dicts."""
    import torch
    _load_glm()

    img    = _render_page(pdf_path, page_num)
    inputs = _glm_processor(
        text   = GLM_TXN_PROMPT,
        images = img,
        return_tensors = "pt",
    ).to(GLM_DEVICE)

    t0 = time.time()
    with torch.no_grad():
        out_ids = _glm_model.generate(
            **inputs,
            max_new_tokens = GLM_MAX_NEW,  # bounded per page — array always closes
            do_sample      = False,
            temperature    = 1.0,
        )
    elapsed   = time.time() - t0
    input_len = inputs["input_ids"].shape[1]
    raw = _glm_processor.decode(
        out_ids[0][input_len:], skip_special_tokens=True
    ).strip()

    out_tokens = out_ids.shape[1] - input_len
    log.info(f"[glm-page-{page_num+1}] {out_tokens} tokens in {elapsed:.1f}s")
    return _extract_json_array(raw, f"glm-page-{page_num+1}")

# ── Public: parse_transactions ─────────────────────────────────────────────────
def parse_transactions(text: str, pdf_path: str = None) -> list:
    """
    Extract ALL transactions using GLM-OCR page-by-page (preferred when pdf_path given).
    Falls back to Qwen chunked text mode only when pdf_path is None.
    """
    if pdf_path:
        return _parse_with_glm(pdf_path)
    log.warning("pdf_path not provided — using Qwen chunked text fallback")
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
        rows   = _clean_amounts(rows)
        before = len(all_txns)
        all_txns = _dedupe(all_txns + rows)
        log.info(
            f"[glm-page-{page_num+1}/{n_pages}]"
            f"  +{len(all_txns)-before} new rows  total={len(all_txns)}"
        )

    log.info(f"[glm-txn] DONE — {len(all_txns)} total transactions")
    return all_txns

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
- date/value_date  : YYYY-MM-DD
- transaction_type : "DEBIT" or "CREDIT"
- amounts          : strip commas and ₹
- Skip header rows
- Include EVERY row — do NOT stop early

BANK STATEMENT TABLE:
"""
_CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "6000"))
_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))

def _parse_with_qwen_fallback(text: str) -> list:
    _load_qwen()
    chunks   = _chunk_text(text, _CHUNK_SIZE, _CHUNK_OVERLAP)
    all_txns: list = []
    log.info(f"[qwen-txn] {len(chunks)} chunks (fallback mode)")
    for i, chunk in enumerate(chunks):
        label = f"qwen-chunk-{i+1}/{len(chunks)}"
        resp  = _qwen.create_chat_completion(
            messages       = [{"role": "user", "content": _QWEN_TXN_PROMPT + chunk}],
            max_tokens     = 4096,   # small per chunk — no overflow
            temperature    = 0.0,
            repeat_penalty = 1.05,
        )
        raw    = resp["choices"][0]["message"]["content"].strip()
        finish = resp["choices"][0].get("finish_reason", "?")
        if finish == "length":
            log.warning(f"[{label}] ⚠️ still cut off — reduce CHUNK_SIZE")
        txns     = _extract_json_array(raw, label)
        all_txns = _dedupe(all_txns + _clean_amounts(txns))
        log.info(f"[{label}] total={len(all_txns)}")
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
        log.error(f"[{label}] JSON decode error: {e}")
        return {}

def _extract_json_array(raw: str, label: str) -> list:
    raw   = _strip_fences(raw)
    start = raw.find("[")
    end   = raw.rfind("]")
    if start == -1 or end <= start:
        log.warning(f"[{label}] No JSON array — 0 rows this page")
        return []
    try:
        result = json.loads(raw[start:end + 1])
        return result if isinstance(result, list) else []
    except json.JSONDecodeError as e:
        log.error(f"[{label}] JSON decode error: {e}\n{raw[start:start+300]}")
        return []

# ── Utilities ──────────────────────────────────────────────────────────────────
def _clean_amounts(items: list) -> list:
    for item in items:
        for f in ("debit", "credit", "balance"):
            if item.get(f):
                item[f] = str(item[f]).replace(",", "").replace("₹", "").strip()
    return items

def _dedupe(transactions: list) -> list:
    seen, out = set(), []
    for txn in transactions:
        key = (
            txn.get("date", ""),
            txn.get("description", "")[:40],
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
