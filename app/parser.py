import os
import re
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("bank_ai.parser")

HF_HOME = os.getenv("HF_HOME", "H:/cache_hub/hub")
os.environ["HF_HOME"]            = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_HOME
os.environ["HF_HUB_CACHE"]       = HF_HOME

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    log.info("HuggingFace login successful")

MODEL_ID   = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
_tokenizer = None
_model     = None


def _load_model():
    global _tokenizer, _model
    if _model is not None:
        log.info("Model already loaded — skipping")
        return

    log.info(f"Loading tokenizer from {MODEL_ID}...")
    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True, cache_dir=HF_HOME
    )
    log.info("Tokenizer loaded ✓")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    log.info("Loading model weights (NF4 4-bit) — allocating VRAM...")
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},        # ← force ALL layers to GPU 0, no CPU offload
        trust_remote_code=True,
        cache_dir=HF_HOME,
        dtype=torch.float16,
    )
    _model.eval()

    # Log VRAM used after load
    free, total = torch.cuda.mem_get_info(0)
    used = round((total - free) / 1e9, 2)
    log.info(f"Model loaded ✓  VRAM used: {used} GB / {round(total/1e9, 2)} GB")


# ── Prompts ───────────────────────────────────────────────────────────────────
HEADER_PROMPT = """You are a bank statement parser. Extract ONLY account/header details from the text below.
Return ONLY valid JSON with this exact schema — leave fields as "" if not found. No explanation.

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

TRANSACTION_PROMPT = """You are a bank transaction extractor. Extract ALL transactions from the text below.
Return ONLY a valid JSON array — no wrapper object, no explanation.

Each item must follow this schema:
{
  "date": "", "value_date": "", "description": "", "narration": "",
  "cheque_no": "", "reference_no": "", "transaction_type": "",
  "debit": "", "credit": "", "balance": "", "branch_code": "", "remarks": ""
}

Rules:
- date / value_date: YYYY-MM-DD (convert from DD/MM/YYYY or DD-MMM-YYYY)
- transaction_type: exactly "DEBIT" or "CREDIT"
- debit and credit are mutually exclusive per row
- amounts: strip symbols/commas — plain number string (e.g. "12500.00")
- Include EVERY transaction row — do not skip any

TEXT:
"""

MAX_HEADER_CHARS  = int(os.getenv("MAX_HEADER_CHARS",  2000))   # header only needs top of doc
MAX_TXN_CHARS     = int(os.getenv("MAX_TXN_CHARS",     6000))   # transactions need more


# ── Inference ─────────────────────────────────────────────────────────────────
def _infer(prompt: str, text_chunk: str, label: str = "inference") -> str:
    import time
    _load_model()

    messages  = [{"role": "user", "content": prompt + text_chunk}]
    formatted = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs     = _tokenizer(formatted, return_tensors="pt").to("cuda")
    input_len  = inputs["input_ids"].shape[1]
    max_tokens = int(os.getenv("MAX_NEW_TOKENS", 2048))

    log.info(f"[{label}] Input tokens: {input_len}  Max new tokens: {max_tokens}")

    t0 = time.time()
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", 4096)),
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
            repetition_penalty=1.15,     # ← penalise repeated tokens, stops "annoy!!!" loops
            temperature=1.0,             # required when do_sample=False
        )
    elapsed = time.time() - t0

    new_tokens = outputs[0][input_len:]
    out_len    = len(new_tokens)
    result     = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    log.info(f"[{label}] Output tokens: {out_len}  Time: {elapsed:.2f}s  "
             f"Speed: {out_len/elapsed:.1f} t/s")
    log.debug(f"[{label}] Raw LLM output (first 500 chars):\n{result[:500]}")

    return result


def _extract_json(raw: str, expect: str = "object", label: str = "") -> any:
    """Strips markdown fences and robustly parses JSON from LLM output."""
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Find the outermost [ ] or { } block
    if expect == "array":
        start, end = raw.find("["), raw.rfind("]")
    else:
        start, end = raw.find("{"), raw.rfind("}")

    if start == -1 or end == -1 or end <= start:
        log.error(f"[{label}] No valid JSON {expect} found in LLM output. Raw:\n{raw[:300]}")
        return [] if expect == "array" else {}

    json_str = raw[start:end + 1]
    try:
        parsed = json.loads(json_str)
        log.info(f"[{label}] JSON parsed successfully — "
                 f"{'items: ' + str(len(parsed)) if expect == 'array' else 'fields: ' + str(len(parsed))}")
        return parsed
    except json.JSONDecodeError as e:
        log.error(f"[{label}] JSON decode error: {e}\nJSON string (first 500):\n{json_str[:500]}")
        return [] if expect == "array" else {}


def parse_header(text: str) -> dict:
    # Only first 2000 chars — header info is always at the top
    chunk = text[:MAX_HEADER_CHARS]
    log.info(f"parse_header: sending {len(chunk):,} chars to LLM")
    raw = _infer(HEADER_PROMPT, chunk, label="header")
    return _extract_json(raw, expect="object", label="header")


def parse_transactions(text: str) -> list:
    # Skip the header section, only send transaction rows
    chunk = text[:MAX_TXN_CHARS]
    log.info(f"parse_transactions: sending {len(chunk):,} / {len(text):,} chars to LLM")
    raw = _infer(TRANSACTION_PROMPT, chunk, label="transactions")
    return _extract_json(raw, expect="array", label="transactions")