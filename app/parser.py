"""
parser.py — HuggingFace Transformers + AutoAWQ inference
Model  : QuantTrio/Qwen3.5-27B-AWQ  (text-only, standard Qwen3 arch, AWQ INT4)
VRAM   : ~17-19 GB on RTX 4090
Backend: transformers + autoawq (no llama-cpp needed)
"""

import os, re, json, logging, time, torch
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("bank_ai.parser")

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID   = os.getenv("MODEL_ID",   "QuantTrio/Qwen3.5-27B-AWQ")
LOCAL_PATH = os.getenv("MODEL_PATH", "/workspace/models/Qwen3.5-27B-AWQ")
HF_TOKEN   = os.getenv("HF_TOKEN")
HF_HOME    = os.getenv("HF_HOME", "/workspace/models")

os.environ["HF_HOME"]            = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_HOME
os.environ["HF_HUB_CACHE"]       = HF_HOME
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MAX_NEW_TOKENS   = int(os.getenv("MAX_NEW_TOKENS",   "8192"))
MAX_HEADER_CHARS = int(os.getenv("MAX_HEADER_CHARS", "4000"))
MAX_TXN_CHARS    = int(os.getenv("MAX_TXN_CHARS",    "30000"))
MAX_CONTEXT      = int(os.getenv("MAX_CONTEXT",      "32768"))

_tokenizer = None
_model     = None


# ── Model loader ───────────────────────────────────────────────────────────────
def _load_model():
    global _tokenizer, _model
    if _model is not None:
        return

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import login

    if HF_TOKEN:
        login(token=HF_TOKEN)
        log.info("HuggingFace login OK")

    load_from = LOCAL_PATH if os.path.exists(
        os.path.join(LOCAL_PATH, "config.json")
    ) else MODEL_ID

    log.info(f"Loading model from: {load_from}")

    # Verify architecture before loading to catch wrong model early
    import json as _json
    cfg_path = os.path.join(load_from, "config.json")
    if os.path.exists(cfg_path):
        cfg = _json.load(open(cfg_path))
        model_type = cfg.get("model_type", "")
        has_vision = "vision" in str(cfg).lower() or "visual" in str(cfg).lower()
        if has_vision:
            raise RuntimeError(
                f"WRONG MODEL: {load_from} is a vision-language model ({model_type}). "
                "Use QuantTrio/Qwen3.5-27B-AWQ (text-only)."
            )
        log.info(f"Architecture check OK — model_type: {model_type}")

    _tokenizer = AutoTokenizer.from_pretrained(
        load_from,
        trust_remote_code=True
    )

    _model = AutoModelForCausalLM.from_pretrained(
        load_from,
        dtype=torch.float16,
        device_map={"": 0},        # all layers → GPU 0
        trust_remote_code=True,
    )
    _model.eval()

    free, total = torch.cuda.mem_get_info(0)
    used = round((total - free) / 1e9, 2)
    tot  = round(total / 1e9, 2)
    log.info(f"Model ready ✓  VRAM used: {used} / {tot} GB")


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

    messages = [{"role": "user", "content": prompt + text}]

    # enable_thinking=False — suppress reasoning block, output pure JSON
    try:
        formatted = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        formatted = _tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs    = _tokenizer(formatted, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    # Safety: truncate if over context window
    max_ctx = MAX_CONTEXT - MAX_NEW_TOKENS - 128
    if input_len > max_ctx:
        ratio = (max_ctx / input_len) * 0.9
        text  = text[:int(len(text) * ratio)]
        log.warning(f"[{label}] Over limit ({input_len} tokens) — truncating to fit context")
        messages = [{"role": "user", "content": prompt + text}]
        try:
            formatted = _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            formatted = _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs    = _tokenizer(formatted, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]

    log.info(f"[{label}] Input: {input_len} tokens | Max new: {MAX_NEW_TOKENS}")
    torch.cuda.empty_cache()

    t0 = time.time()
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens     = MAX_NEW_TOKENS,
            do_sample          = False,
            repetition_penalty = 1.05,
            pad_token_id       = _tokenizer.eos_token_id,
            eos_token_id       = _tokenizer.eos_token_id,
        )

    elapsed    = time.time() - t0
    new_tokens = outputs[0][input_len:]
    raw_result = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    result     = _strip_thinking(raw_result)

    tps = len(new_tokens) / max(elapsed, 0.1)
    log.info(f"[{label}] {len(new_tokens)} tokens in {elapsed:.2f}s ({tps:.0f} t/s)")
    return result


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
    """Pre-load model into VRAM at startup (called from main.py lifespan)."""
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
