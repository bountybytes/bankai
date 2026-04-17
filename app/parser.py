"""
parser.py — llama-cpp-python GGUF inference
Model  : Qwen2.5-Coder-14B-Instruct Q4_K_M
Strategy: table-structured input → LLM only needs to reformat, not search
"""

import os, re, json, logging, time
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("bank_ai.parser")

MODEL_PATH       = os.getenv("MODEL_PATH",       "/workspace/models/qwen2.5-coder-14b-instruct-q4_k_m.gguf")
N_GPU_LAYERS     = int(os.getenv("N_GPU_LAYERS",   "-1"))
N_CTX            = int(os.getenv("N_CTX",          "65536"))
MAX_NEW_TOKENS   = int(os.getenv("MAX_NEW_TOKENS",  "16384"))
MAX_HEADER_CHARS = int(os.getenv("MAX_HEADER_CHARS","6000"))
MAX_TXN_CHARS    = int(os.getenv("MAX_TXN_CHARS",   "48000"))
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE",      "10000"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP",   "300"))

_llm = None


def _load_model():
    global _llm
    if _llm is not None:
        return
    from llama_cpp import Llama
    log.info(f"Loading GGUF: {MODEL_PATH}")
    log.info(f"  n_ctx={N_CTX}  n_gpu_layers={N_GPU_LAYERS}  max_new={MAX_NEW_TOKENS}")
    _llm = Llama(
        model_path   = MODEL_PATH,
        n_ctx        = N_CTX,
        n_gpu_layers = N_GPU_LAYERS,
        n_threads    = 8,
        flash_attn   = True,
        verbose      = False,
    )
    log.info("GGUF model ready ✓")


HEADER_PROMPT = """You are a precise JSON extractor for Indian bank statements.
Extract ONLY account/header details from the text below.
Output ONLY a valid JSON object. No explanation. No markdown. No extra text.

Exact schema (use "" for missing):
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
- statement_from / statement_to: YYYY-MM-DD
- opening_balance / closing_balance: plain number NO commas e.g. "21675.91"
- currency: "INR" if implied

TEXT:
"""

TRANSACTION_PROMPT = """You are a precise JSON extractor for Indian bank transactions.
The input is pipe-delimited table rows from a bank statement PDF.
Extract EVERY transaction row. Output ONLY a valid JSON array. No explanation. No markdown.

Each object must use this exact schema:
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
- date / value_date: YYYY-MM-DD (convert DD/MM/YYYY, DD-MM-YYYY, DD-MMM-YYYY)
- transaction_type: exactly "DEBIT" or "CREDIT" — if debit column has value use "DEBIT", credit column use "CREDIT"
- debit / credit: mutually exclusive — one must be "" per row
- ALL amounts: strip commas and ₹ symbols e.g. "12500.00" NOT "12,500.00"
- balance: plain number e.g. "1268.80" NOT "1,268.80"
- Skip header rows (rows containing "Date", "Narration", "Debit", "Credit", "Balance" etc.)
- Include EVERY data row — do NOT stop early
- use "" for absent fields

BANK STATEMENT TABLE:
"""


def _infer(prompt: str, text: str, label: str = "") -> str:
    _load_model()

    full_prompt   = prompt + text
    approx_tokens = len(full_prompt) // 4
    max_input     = N_CTX - MAX_NEW_TOKENS - 512

    if approx_tokens > max_input:
        ratio       = (max_input / approx_tokens) * 0.92
        text        = text[:int(len(text) * ratio)]
        full_prompt = prompt + text
        log.warning(f"[{label}] Truncated: ~{approx_tokens} → ~{len(full_prompt)//4} tokens")

    log.info(f"[{label}] Sending ~{len(full_prompt)//4} tokens | Max new: {MAX_NEW_TOKENS}")

    t0 = time.time()
    response = _llm.create_chat_completion(
        messages      = [{"role": "user", "content": full_prompt}],
        max_tokens    = MAX_NEW_TOKENS,
        temperature   = 0.0,
        repeat_penalty= 1.05,
    )
    elapsed    = time.time() - t0
    raw        = response["choices"][0]["message"]["content"].strip()
    out_tokens = response.get("usage", {}).get("completion_tokens", 0)
    finish     = response["choices"][0].get("finish_reason", "?")
    log.info(f"[{label}] {out_tokens} tokens in {elapsed:.2f}s ({int(out_tokens)/max(elapsed,0.1):.0f} t/s) finish={finish}")
    if finish == "length":
        log.warning(f"[{label}] ⚠️  Output CUT OFF — increase MAX_NEW_TOKENS or reduce chunk size")
    return raw


def _extract_json(raw: str, expect: str = "object", label: str = ""):
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*",     "", raw).strip()
    open_c  = "[" if expect == "array" else "{"
    close_c = "]" if expect == "array" else "}"
    start   = raw.find(open_c)
    end     = raw.rfind(close_c)
    if start == -1 or end <= start:
        log.error(f"[{label}] No valid JSON {expect}.\nRaw:\n{raw[:600]}")
        return [] if expect == "array" else {}
    try:
        parsed = json.loads(raw[start:end + 1])
        count  = len(parsed) if isinstance(parsed, list) else len(parsed.keys())
        log.info(f"[{label}] Parsed OK — {'items: '+str(count) if expect=='array' else 'fields: '+str(count)}")
        return parsed
    except json.JSONDecodeError as e:
        log.error(f"[{label}] JSON error: {e}\n{raw[start:start+500]}")
        return [] if expect == "array" else {}


def _clean_amounts(items) -> list:
    amount_fields = {"debit", "credit", "balance", "opening_balance", "closing_balance"}
    for item in items:
        for f in amount_fields:
            if f in item and item[f]:
                item[f] = str(item[f]).replace(",", "").replace("₹", "").strip()
    return items


def _dedupe(transactions: list) -> list:
    seen, out = set(), []
    for txn in transactions:
        key = f"{txn.get('date')}-{txn.get('description','')[:40]}-{txn.get('debit')}-{txn.get('credit')}"
        if key not in seen:
            seen.add(key)
            out.append(txn)
    return out


def _chunk_text(text: str) -> list:
    chunks, start = [], 0
    while start < len(text):
        end        = start + CHUNK_SIZE
        if end < len(text):
            nl = text.rfind("\n", start + CHUNK_SIZE - CHUNK_OVERLAP, end)
            if nl != -1:
                end = nl
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP if end < len(text) else len(text)
    return chunks


def load_model():
    _load_model()


def parse_header(text: str) -> dict:
    raw    = _infer(HEADER_PROMPT, text[:MAX_HEADER_CHARS], label="header")
    result = _extract_json(raw, expect="object", label="header")
    for f in ("opening_balance", "closing_balance"):
        if result.get(f):
            result[f] = str(result[f]).replace(",", "").replace("₹", "").strip()
    return result


def parse_transactions(text: str) -> list:
    if len(text) <= MAX_TXN_CHARS:
        log.info(f"Transactions: single-shot {len(text):,} chars")
        raw  = _infer(TRANSACTION_PROMPT, text, label="transactions")
        txns = _extract_json(raw, expect="array", label="transactions")
        return _dedupe(_clean_amounts(txns))

    chunks   = _chunk_text(text)
    all_txns = []
    log.info(f"Transactions: chunked {len(text):,} chars → {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        label = f"txn-chunk-{i+1}/{len(chunks)}"
        raw   = _infer(TRANSACTION_PROMPT, chunk, label=label)
        txns  = _extract_json(raw, expect="array", label=label)
        txns  = _clean_amounts(txns)
        before = len(all_txns)
        all_txns = _dedupe(all_txns + txns)
        log.info(f"[{label}] +{len(all_txns)-before} new rows (total: {len(all_txns)})")

    return all_txns
