"""
parser.py — GGUF inference via llama-cpp-python
Model : Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF
Quant : Q4_K_M  (~17 GB on disk, ~22 GB VRAM on 4090)
Backend: llama-cpp-python with CUDA (n_gpu_layers=-1 = all layers on GPU)

No token output limits — context window is 32 768 tokens; output can fill
whatever remains after the prompt (up to ~28 000 new tokens if needed).
"""

import os, re, json, logging, time
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("bank_ai.parser")

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH   = os.getenv(
    "GGUF_MODEL_PATH",
    "/workspace/models/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-Q4_K_M.gguf"
)
N_CTX        = int(os.getenv("N_CTX",        "32768"))   # full context window
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", "-1"))       # -1 = all layers → GPU
N_THREADS    = int(os.getenv("N_THREADS",    "8"))
TEMPERATURE  = float(os.getenv("TEMPERATURE","0.0"))      # deterministic for JSON

# No hard cap — let the model finish naturally.
# 0 means "use whatever fits in the remaining context window"
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "0"))

# Max chars sent to the model per call
MAX_HEADER_CHARS = int(os.getenv("MAX_HEADER_CHARS", "4000"))
MAX_TXN_CHARS    = int(os.getenv("MAX_TXN_CHARS",    "30000"))

_llm = None   # global model handle


# ── Model loader ───────────────────────────────────────────────────────────────
def _load_model():
    global _llm
    if _llm is not None:
        return

    from llama_cpp import Llama

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"GGUF model not found at: {MODEL_PATH}\n"
            "Download with:\n"
            "  huggingface-cli download Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF "
            "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-Q4_K_M.gguf "
            "--local-dir /workspace/models"
        )

    log.info(f"Loading GGUF model: {MODEL_PATH}")
    log.info(f"  n_ctx={N_CTX}  n_gpu_layers={N_GPU_LAYERS}  temp={TEMPERATURE}")

    _llm = Llama(
        model_path    = MODEL_PATH,
        n_ctx         = N_CTX,
        n_gpu_layers  = N_GPU_LAYERS,   # all layers on VRAM
        n_threads     = N_THREADS,
        verbose       = False,
        use_mlock     = True,           # pin model weights in RAM, no swapping
        flash_attn    = True,           # FlashAttention-2 — faster + less VRAM
    )
    log.info("GGUF model ready ✓")


# ── Prompts ────────────────────────────────────────────────────────────────────
HEADER_PROMPT = """\
You are a bank statement parser. Extract ONLY account/header details from the text below.
Return ONLY valid JSON with this exact schema — leave fields as "" if not found. No explanation, no markdown.

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
Return ONLY a valid JSON array — no wrapper object, no markdown, no explanation.

Each item must follow this exact schema:
{
  "date": "", "value_date": "", "description": "", "narration": "",
  "cheque_no": "", "reference_no": "", "transaction_type": "",
  "debit": "", "credit": "", "balance": "", "branch_code": "", "remarks": ""
}

Rules:
- date / value_date: YYYY-MM-DD (convert from DD/MM/YYYY or DD-MMM-YYYY)
- transaction_type: exactly "DEBIT" or "CREDIT" — infer from context if column is missing
- debit and credit are mutually exclusive per row
- amounts: strip all currency symbols and commas — plain number string (e.g. "12500.00")
- Include EVERY transaction row — do NOT skip any, including opening/closing balance rows
- Output must be a single JSON array starting with [ and ending with ]

TEXT:
"""


# ── Core inference ─────────────────────────────────────────────────────────────
def _infer(prompt: str, text: str, label: str = "") -> str:
    _load_model()

    full_prompt = (
        "<|im_start|>system\n"
        "You are a precise financial data extraction AI. Always return valid JSON only.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}{text}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # Calculate safe max tokens — fill remaining context window
    # Rough estimate: 1 token ≈ 3 chars
    prompt_tokens    = len(full_prompt) // 3
    remaining_tokens = N_CTX - prompt_tokens - 64
    max_tokens       = MAX_NEW_TOKENS if MAX_NEW_TOKENS > 0 else max(remaining_tokens, 1024)

    log.info(f"[{label}] Prompt ~{prompt_tokens} tokens | Max new tokens: {max_tokens}")

    t0 = time.time()
    output = _llm(
        full_prompt,
        max_tokens   = max_tokens,
        temperature  = TEMPERATURE,
        repeat_penalty = 1.05,
        stop         = ["<|im_end|>", "<|endoftext|>"],
        echo         = False,
    )
    elapsed = time.time() - t0

    result      = output["choices"][0]["text"].strip()
    tokens_used = output.get("usage", {}).get("completion_tokens", len(result) // 3)
    log.info(f"[{label}] Done in {elapsed:.2f}s | ~{tokens_used} output tokens | {tokens_used/max(elapsed,0.1):.0f} t/s")
    return result


# ── JSON extraction ────────────────────────────────────────────────────────────
def _extract_json(raw: str, expect: str = "object", label: str = "") -> any:
    # Strip markdown fences if the model added them
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*",     "", raw)
    raw = raw.strip()

    open_char  = "[" if expect == "array" else "{"
    close_char = "]" if expect == "array" else "}"

    start = raw.find(open_char)
    end   = raw.rfind(close_char)

    if start == -1 or end <= start:
        log.error(f"[{label}] No valid JSON {expect} in output.\nRaw (first 600):\n{raw[:600]}")
        return [] if expect == "array" else {}

    snippet = raw[start:end + 1]
    try:
        parsed = json.loads(snippet)
        count  = len(parsed) if isinstance(parsed, list) else len(parsed.keys())
        log.info(f"[{label}] ✓ Parsed — {'items: ' + str(count) if expect == 'array' else 'fields: ' + str(count)}")
        return parsed
    except json.JSONDecodeError as e:
        log.error(f"[{label}] JSON decode error: {e}\nSnippet start:\n{snippet[:500]}")
        return [] if expect == "array" else {}


# ── Public interface ───────────────────────────────────────────────────────────
def load_model():
    """Called at app startup to pre-load the model into VRAM."""
    _load_model()


def parse_header(text: str) -> dict:
    raw = _infer(HEADER_PROMPT, text[:MAX_HEADER_CHARS], label="header")
    return _extract_json(raw, expect="object", label="header")


def parse_transactions(text: str) -> list:
    chunk = text[:MAX_TXN_CHARS]
    log.info(f"Transactions: sending {len(chunk):,} / {len(text):,} chars to LLM")
    raw = _infer(TRANSACTION_PROMPT, chunk, label="transactions")
    return _extract_json(raw, expect="array", label="transactions")