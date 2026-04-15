"""
parser.py — llama-cpp-python inference (GGUF)
Model  : Qwen3.5-27B.Q4_K_M.gguf
VRAM   : ~17-19 GB on RTX 4090
Backend: llama-cpp-python with CUDA offload (no transformers/AWQ needed)
"""

import os, re, json, logging, time
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("bank_ai.parser")

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH     = os.getenv("MODEL_PATH",     "/workspace/models/Qwen3.5-27B-Q4_K_M.gguf")
N_GPU_LAYERS   = int(os.getenv("N_GPU_LAYERS",   "-1"))   # -1 = all layers on GPU
N_CTX          = int(os.getenv("N_CTX",          "32768"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS",  "8192"))
MAX_HEADER_CHARS = int(os.getenv("MAX_HEADER_CHARS", "4000"))
MAX_TXN_CHARS    = int(os.getenv("MAX_TXN_CHARS",    "30000"))

_llm = None


# ── Model loader ───────────────────────────────────────────────────────────────
def _load_model():
    global _llm
    if _llm is not None:
        return

    from llama_cpp import Llama

    log.info(f"Loading GGUF model: {MODEL_PATH}")
    log.info(f"  n_ctx={N_CTX}  n_gpu_layers={N_GPU_LAYERS}  temp=0.0")

    _llm = Llama(
        model_path    = MODEL_PATH,
        n_ctx         = N_CTX,
        n_gpu_layers  = N_GPU_LAYERS,
        n_threads     = 8,
        flash_attn    = False,       # disable — safer on first run
        verbose       = False,
    )
    log.info("GGUF model ready ✓")


# ── Prompts ────────────────────────────────────────────────────────────────────
HEADER_PROMPT = """\
You are a bank statement parser. Extract ONLY account/header details from the text below.
Return ONLY valid JSON with this exact schema — leave fields as "" if not found.
No explanation. No markdown. Just the JSON object.

{
  "bank_name": "", "branch_name": "", "bank_type": "",
  "account_number": "", "account_holder": "", "address": "",
  "pan": "", "mobile": "", "email": "",
  "ifsc_code": "", "micr_code": "", "account_type": "",
  "statement_from": "", "statement_to": "",
  "opening_balance": "", "closing_balance": "", "currency": ""
}

Rules:
- statement_from / statement_to: YYYY-MM-DD format
- opening_balance / closing_balance: plain number string, no symbols (e.g. "12500.00")

TEXT:
"""

TRANSACTION_PROMPT = """\
You are a bank transaction extractor. Extract ALL transactions from the text below.
Return ONLY a valid JSON array — no wrapper object, no explanation, no markdown.

Each item must follow this exact schema:
{
  "date": "", "value_date": "", "description": "", "narration": "",
  "cheque_no": "", "reference_no": "", "transaction_type": "",
  "debit": "", "credit": "", "balance": "", "branch_code": "", "remarks": ""
}

Rules:
- date / value_date: YYYY-MM-DD (convert from DD/MM/YYYY or DD-MMM-YYYY)
- transaction_type: exactly "DEBIT" or "CREDIT" — infer from context if column missing
- debit and credit are mutually exclusive per row
- amounts: strip all currency symbols and commas — plain number string (e.g. "12500.00")
- Include EVERY transaction row — do NOT skip any

TEXT:
"""


# ── Strip <think>...</think> blocks (Qwen3 thinking mode) ─────────────────────
def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"</think>", "", text)
    return text.strip()


# ── Core inference ─────────────────────────────────────────────────────────────
def _infer(prompt: str, text: str, label: str = "") -> str:
    _load_model()

    full_prompt = prompt + text

    # Approximate token count (4 chars ≈ 1 token)
    approx_tokens = len(full_prompt) // 4
    max_input     = N_CTX - MAX_NEW_TOKENS - 128

    if approx_tokens > max_input:
        ratio = (max_input / approx_tokens) * 0.9
        text  = text[:int(len(text) * ratio)]
        log.warning(f"[{label}] Estimated over limit — truncating text to {len(text):,} chars")
        full_prompt = prompt + text

    log.info(f"[{label}] Sending ~{len(full_prompt)//4} tokens | Max new: {MAX_NEW_TOKENS}")

    t0 = time.time()
    response = _llm.create_chat_completion(
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens    = MAX_NEW_TOKENS,
        temperature   = 0.0,
        repeat_penalty= 1.05,
    )
    elapsed = time.time() - t0

    raw = response["choices"][0]["message"]["content"].strip()

    # Usage stats
    usage = response.get("usage", {})
    out_tokens = usage.get("completion_tokens", "?")
    log.info(f"[{label}] {out_tokens} tokens in {elapsed:.2f}s ({int(out_tokens)/max(elapsed,0.1):.0f} t/s)")

    return _strip_thinking(raw)


# ── JSON extraction ────────────────────────────────────────────────────────────
def _extract_json(raw: str, expect: str = "object", label: str = "") -> any:
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*",     "", raw).strip()

    open_c  = "[" if expect == "array" else "{"
    close_c = "]" if expect == "array" else "}"
    start   = raw.find(open_c)
    end     = raw.rfind(close_c)

    if start == -1 or end <= start:
        log.error(f"[{label}] No valid JSON {expect}.\nRaw (first 600):\n{raw[:600]}")
        return [] if expect == "array" else {}

    try:
        parsed = json.loads(raw[start:end + 1])
        count  = len(parsed) if isinstance(parsed, list) else len(parsed.keys())
        log.info(f"[{label}] ✓ Parsed — {'items: '+str(count) if expect=='array' else 'fields: '+str(count)}")
        return parsed
    except json.JSONDecodeError as e:
        log.error(f"[{label}] JSON decode error: {e}\n{raw[start:start+500]}")
        return [] if expect == "array" else {}


# ── Public interface ───────────────────────────────────────────────────────────
def load_model():
    """Pre-load GGUF model into VRAM at startup (called from main.py lifespan)."""
    _load_model()


def parse_header(text: str) -> dict:
    """Extract account/header details from raw statement text."""
    raw = _infer(HEADER_PROMPT, text[:MAX_HEADER_CHARS], label="header")
    return _extract_json(raw, expect="object", label="header")


def parse_transactions(text: str) -> list:
    """Extract all transaction rows from raw statement text."""
    chunk = text[:MAX_TXN_CHARS]
    log.info(f"Transactions: sending {len(chunk):,} / {len(text):,} chars to LLM")
    raw = _infer(TRANSACTION_PROMPT, chunk, label="transactions")
    return _extract_json(raw, expect="array", label="transactions")
