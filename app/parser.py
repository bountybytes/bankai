"""
parser.py — llama-cpp-python inference (GGUF)
Model  : Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf
VRAM   : ~9-10 GB on RTX 4090
Backend: llama-cpp-python with CUDA offload
"""

import os, re, json, logging, time
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("bank_ai.parser")

MODEL_PATH       = os.getenv("MODEL_PATH",       "/workspace/models/qwen2.5-coder-14b-instruct-q4_k_m.gguf")
N_GPU_LAYERS     = int(os.getenv("N_GPU_LAYERS",   "-1"))
N_CTX            = int(os.getenv("N_CTX",          "65536"))
MAX_NEW_TOKENS   = int(os.getenv("MAX_NEW_TOKENS",  "16384"))
MAX_HEADER_CHARS = int(os.getenv("MAX_HEADER_CHARS","8000"))
MAX_TXN_CHARS    = int(os.getenv("MAX_TXN_CHARS",   "48000"))
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE",      "12000"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP",   "500"))

_llm = None


def _load_model():
    global _llm
    if _llm is not None:
        return

    from llama_cpp import Llama

    log.info(f"Loading GGUF model: {MODEL_PATH}")
    log.info(f"  n_ctx={N_CTX}  n_gpu_layers={N_GPU_LAYERS}")

    _llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        n_threads=8,
        flash_attn=True,
        verbose=False,
    )
    log.info("GGUF model ready ✓")


HEADER_PROMPT = """You are a precise JSON extractor for Indian bank statements.
Extract ONLY account/header details from the bank statement text below.
Output ONLY a valid JSON object. No explanation. No markdown. No extra text.

Return exactly this schema:
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
- statement_from and statement_to must be YYYY-MM-DD
- opening_balance and closing_balance must be plain number strings
- use "" when missing
- currency should be "INR" if implied but not printed

TEXT:
"""

TRANSACTION_PROMPT = """You are a precise JSON extractor for Indian bank statements.
Extract ALL transactions from the bank statement text below.
Output ONLY a valid JSON array. No explanation. No markdown. No extra text.

Each item must follow this exact schema:
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
- date and value_date must be YYYY-MM-DD
- convert DD/MM/YYYY, DD-MM-YYYY, DD-MMM-YYYY to YYYY-MM-DD
- transaction_type must be exactly "DEBIT" or "CREDIT"
- debit and credit are mutually exclusive
- amounts must be plain number strings without commas or symbols
- include every transaction row
- if a field is absent, use ""

TEXT:
"""


def _infer(prompt: str, text: str, label: str = "") -> str:
    _load_model()

    full_prompt = prompt + text
    approx_tokens = len(full_prompt) // 4
    max_input = N_CTX - MAX_NEW_TOKENS - 256

    if approx_tokens > max_input:
        ratio = (max_input / approx_tokens) * 0.9
        text = text[:int(len(text) * ratio)]
        full_prompt = prompt + text
        log.warning(f"[{label}] Truncated input to fit context")

    log.info(f"[{label}] Sending ~{len(full_prompt)//4} tokens | Max new: {MAX_NEW_TOKENS}")

    t0 = time.time()
    response = _llm.create_chat_completion(
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        repeat_penalty=1.05,
    )
    elapsed = time.time() - t0

    raw = response["choices"][0]["message"]["content"].strip()
    usage = response.get("usage", {})
    out_tokens = usage.get("completion_tokens", 0)
    log.info(f"[{label}] {out_tokens} tokens in {elapsed:.2f}s")
    return raw


def _extract_json(raw: str, expect: str = "object", label: str = ""):
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw).strip()

    open_c = "[" if expect == "array" else "{"
    close_c = "]" if expect == "array" else "}"
    start = raw.find(open_c)
    end = raw.rfind(close_c)

    if start == -1 or end <= start:
        log.error(f"[{label}] No valid JSON found. Raw: {raw[:600]}")
        return [] if expect == "array" else {}

    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError as e:
        log.error(f"[{label}] JSON decode error: {e}")
        return [] if expect == "array" else {}


def load_model():
    _load_model()


def parse_header(text: str) -> dict:
    raw = _infer(HEADER_PROMPT, text[:MAX_HEADER_CHARS], label="header")
    return _extract_json(raw, expect="object", label="header")


def parse_transactions(text: str) -> list:
    chunk = text[:MAX_TXN_CHARS]
    log.info(f"Transactions: sending {len(chunk):,} / {len(text):,} chars to LLM")
    raw = _infer(TRANSACTION_PROMPT, chunk, label="transactions")
    return _extract_json(raw, expect="array", label="transactions")
