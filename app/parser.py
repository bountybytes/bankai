import os, re, json, logging, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("bank_ai.parser")

HF_HOME  = os.getenv("HF_HOME", "/workspace/models")
MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

os.environ["HF_HOME"]            = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_HOME
os.environ["HF_HUB_CACHE"]       = HF_HOME

_tokenizer = None
_model     = None

def _load_model():
    global _tokenizer, _model
    if _model is not None:
        return

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        log.info("HuggingFace login OK")

    log.info(f"Downloading / loading: {MODEL_ID}")
    log.info("This will take ~3–5 min on first run (downloading ~15GB)...")

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        cache_dir=HF_HOME,
    )

    # Full FP16 — no quantization on 24GB GPU
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True,
        cache_dir=HF_HOME,
    )
    _model.eval()

    free, total = torch.cuda.mem_get_info(0)
    log.info(f"Model ready ✓  VRAM: {round((total-free)/1e9,2)} / {round(total/1e9,2)} GB used")

MAX_CONTEXT      = int(os.getenv("MAX_CONTEXT",      7168))
MAX_NEW_TOKENS   = int(os.getenv("MAX_NEW_TOKENS",   3000))
MAX_HEADER_CHARS = int(os.getenv("MAX_HEADER_CHARS", 2000))
MAX_TXN_CHARS    = int(os.getenv("MAX_TXN_CHARS",   10000))

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
- opening_balance / closing_balance: plain number string (e.g. "12500.00")

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
- amounts: strip symbols/commas (e.g. "12500.00")
- Include EVERY transaction row — do not skip any

TEXT:
"""

def _infer(prompt: str, text_chunk: str, label: str = "") -> str:
    _load_model()
    messages  = [{"role": "user", "content": prompt + text_chunk}]
    formatted = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs    = _tokenizer(formatted, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    # Guard against context overflow
    max_input = MAX_CONTEXT - MAX_NEW_TOKENS - 64
    if input_len > max_input:
        log.warning(f"[{label}] Truncating {input_len} → {max_input} tokens")
        ratio      = (max_input / input_len) * 0.9
        text_chunk = text_chunk[:int(len(text_chunk) * ratio)]
        messages   = [{"role": "user", "content": prompt + text_chunk}]
        formatted  = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs     = _tokenizer(formatted, return_tensors="pt").to("cuda")
        input_len  = inputs["input_ids"].shape[1]

    log.info(f"[{label}] Input: {input_len} tokens  Max new: {MAX_NEW_TOKENS}")
    torch.cuda.empty_cache()

    t0 = time.time()
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=_tokenizer.eos_token_id,
        )
    elapsed    = time.time() - t0
    new_tokens = outputs[0][input_len:]
    result     = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    log.info(f"[{label}] {len(new_tokens)} tokens in {elapsed:.2f}s = {len(new_tokens)/max(elapsed,0.1):.0f} t/s")
    return result

def _extract_json(raw: str, expect: str = "object", label: str = "") -> any:
    raw   = raw.replace("```json", "").replace("```", "").strip()
    start = raw.find("[" if expect == "array" else "{")
    end   = raw.rfind("]" if expect == "array" else "}")
    if start == -1 or end <= start:
        log.error(f"[{label}] No valid JSON {expect} found.\nRaw:\n{raw[:400]}")
        return [] if expect == "array" else {}
    try:
        parsed = json.loads(raw[start:end+1])
        log.info(f"[{label}] Parsed OK — {'items: '+str(len(parsed)) if expect=='array' else 'fields: '+str(len(parsed))}")
        return parsed
    except json.JSONDecodeError as e:
        log.error(f"[{label}] JSON error: {e}\n{raw[start:end+1][:400]}")
        return [] if expect == "array" else {}

def parse_header(text: str) -> dict:
    raw = _infer(HEADER_PROMPT, text[:MAX_HEADER_CHARS], label="header")
    return _extract_json(raw, expect="object", label="header")

def parse_transactions(text: str) -> list:
    chunk = text[:MAX_TXN_CHARS]
    log.info(f"Transactions: {len(chunk):,} / {len(text):,} chars")
    raw = _infer(TRANSACTION_PROMPT, chunk, label="transactions")
    return _extract_json(raw, expect="array", label="transactions")
