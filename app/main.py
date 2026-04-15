"""
main.py — FastAPI application for Bank Statement AI
Model : Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF (Q4_K_M)
Run   : python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

import os
import tempfile
import time
import logging
import uuid
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from app.extractor    import extract_text, anonymize_sensitive, restore_sensitive
from app.parser       import parse_header, parse_transactions, load_model
from app.categorizer  import categorize_transactions
from app.schemas      import ParseResponse

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bank_ai.main")


# ── Lifespan — load model at startup ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 60)
    log.info("STARTUP  — Bank Statement AI (GGUF)")
    log.info(f"  Model  : {os.getenv('GGUF_MODEL_PATH', 'Qwen3.5-27B Q4_K_M')}")
    log.info(f"  n_ctx  : {os.getenv('N_CTX', '32768')}")
    log.info("Loading GGUF model into VRAM (~22 GB) — first run takes ~30s ...")
    load_model()
    log.info("Model ready ✓  Server is now accepting requests.")
    log.info("=" * 60)
    yield
    log.info("SHUTDOWN — releasing GPU resources.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Bank Statement AI",
    description = """
## Automated Bank Statement Parser — powered by Qwen3.5-27B (GGUF Q4_K_M)

Upload any digital Indian bank statement PDF and receive:

- ✅ **Structured account details** — holder, account number, IFSC, PAN, branch
- ✅ **All transactions** — dates, amounts, cheque/reference numbers, running balance
- ✅ **Auto-assigned spending categories** — 21 categories via regex rules
- ✅ **Financial summary** — total debits, credits, net flow
- ✅ **Pipeline timings** — PDF extraction, LLM inference, categorisation

### Supported Banks
Axis Bank · HDFC · SBI · ICICI · Kotak · Yes Bank · IndusInd · PNB · Bank of Baroda · Canara · Federal Bank · and more

### Privacy
All processing is **100% local** — no data sent to any third-party API.
PAN numbers and mobile numbers are anonymised before reaching the LLM.
""",
    version     = "2.0.0",
    contact     = {"name": "Rahul Hrushikesh"},
    license_info= {"name": "MIT"},
    openapi_tags= [
        {"name": "Parsing", "description": "Core PDF upload and AI extraction endpoints."},
        {"name": "System",  "description": "Health check, CUDA status, and server diagnostics."},
    ],
    lifespan    = lifespan,
)


# ── POST /parse ────────────────────────────────────────────────────────────────
@app.post(
    "/parse",
    tags            = ["Parsing"],
    response_model  = ParseResponse,
    summary         = "Parse Bank Statement PDF",
    description     = """
Upload a bank statement PDF. The pipeline runs in 4 stages:

1. **PDF Extraction** — PyMuPDF extracts raw text (~0.1s)
2. **Anonymisation** — PAN / mobile numbers redacted before LLM sees them
3. **LLM Inference** — Qwen3.5-27B GGUF parses header + all transactions (~15–40s)
4. **Categorisation** — Regex rules assign spending categories (~1ms)
""",
    responses       = {
        200: {"description": "Successfully parsed bank statement"},
        400: {"description": "Invalid file — only PDF accepted",
              "content": {"application/json": {"example": {"detail": "Only PDF files supported"}}}},
        500: {"description": "LLM inference or extraction failed"},
    },
)
async def parse_statement(
    file: UploadFile = File(..., description="Bank statement PDF (digital, not scanned)")
):
    req_id = uuid.uuid4().hex[:8]
    log.info(f"[{req_id}] ── NEW REQUEST ──────────────────────────────")
    log.info(f"[{req_id}] File : {file.filename}  ({file.content_type})")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    t0 = time.time()

    # ── Stage 1: PDF extraction ────────────────────────────────────────────────
    log.info(f"[{req_id}] Stage 1 — PDF extraction")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        raw_text = extract_text(tmp_path)
        os.unlink(tmp_path)
        log.info(f"[{req_id}]   Extracted {len(raw_text):,} chars from {len(content)/1024:.1f} KB PDF")
    except Exception as e:
        log.error(f"[{req_id}] Stage 1 FAILED: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF extraction error: {e}")
    t1 = time.time()
    log.info(f"[{req_id}]   ✓ {t1 - t0:.3f}s")

    # ── Stage 2: Anonymise sensitive fields ────────────────────────────────────
    anon_text, restore_map = anonymize_sensitive(raw_text)
    log.info(f"[{req_id}] Stage 2 — Anonymised {len(restore_map)} sensitive fields")

    # ── Stage 3: LLM inference ─────────────────────────────────────────────────
    log.info(f"[{req_id}] Stage 3 — LLM inference")
    try:
        log.info(f"[{req_id}]   Parsing header...")
        acc_details  = parse_header(anon_text)
        acc_details  = restore_sensitive(acc_details, restore_map)

        log.info(f"[{req_id}]   Header  → account: {acc_details.get('account_number', 'N/A')}  holder: {acc_details.get('account_holder', 'N/A')}")

        log.info(f"[{req_id}]   Parsing transactions ({len(anon_text):,} chars)...")
        transactions = parse_transactions(anon_text)
        transactions = [restore_sensitive(t, restore_map) for t in transactions]

        log.info(f"[{req_id}]   Transactions → {len(transactions)} rows found")
    except Exception as e:
        log.error(f"[{req_id}] Stage 3 FAILED: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM inference error: {e}")
    t2 = time.time()
    log.info(f"[{req_id}]   ✓ {t2 - t1:.3f}s")

    # ── Stage 4: Categorisation ────────────────────────────────────────────────
    log.info(f"[{req_id}] Stage 4 — Categorising {len(transactions)} transactions")
    transactions = categorize_transactions(transactions)
    cats = Counter(t.get("category", "Other") for t in transactions)
    for cat, count in cats.most_common():
        log.info(f"[{req_id}]   {cat:<30} {count} txns")
    t3 = time.time()
    log.info(f"[{req_id}]   ✓ {(t3 - t2) * 1000:.2f}ms")

    # ── Build summary ──────────────────────────────────────────────────────────
    debits, credits = [], []
    for txn in transactions:
        try:
            if txn.get("debit")  and str(txn["debit"]).strip():
                debits.append(float(txn["debit"]))
            if txn.get("credit") and str(txn["credit"]).strip():
                credits.append(float(txn["credit"]))
        except (ValueError, TypeError):
            pass

    summary = {
        "total_transactions": len(transactions),
        "total_debit":        round(sum(debits),   2),
        "total_credit":       round(sum(credits),  2),
        "net_flow":           round(sum(credits) - sum(debits), 2),
    }

    total_s = round(time.time() - t0, 3)
    log.info(f"[{req_id}] ── COMPLETE ─────────────────────────────────")
    log.info(f"[{req_id}] Transactions : {summary['total_transactions']}")
    log.info(f"[{req_id}] Total Debit  : ₹{summary['total_debit']:,.2f}")
    log.info(f"[{req_id}] Total Credit : ₹{summary['total_credit']:,.2f}")
    log.info(f"[{req_id}] Net Flow     : ₹{summary['net_flow']:,.2f}")
    log.info(f"[{req_id}] Total Time   : {total_s}s  (pdf={t1-t0:.3f}s  llm={t2-t1:.3f}s  cat={(t3-t2)*1000:.2f}ms)")

    return JSONResponse(content={
        "account_details":  acc_details,
        "transactions":     transactions,
        "summary":          summary,
        "source_file":      file.filename,
        "used_image_mode":  False,
        "timings": {
            "pdf_extraction_s": round(t1 - t0,  3),
            "llm_inference_s":  round(t2 - t1,  3),
            "categorization_ms":round((t3 - t2) * 1000, 2),
            "total_s":          total_s,
        },
    })


# ── GET /health ────────────────────────────────────────────────────────────────
@app.get(
    "/health",
    tags        = ["System"],
    summary     = "Server Health Check",
    description = "Returns CUDA availability, GPU name, free VRAM, and model load status.",
)
def health():
    import torch
    from app.parser import _llm

    cuda_ok    = torch.cuda.is_available()
    vram_free  = vram_total = vram_used = 0.0
    if cuda_ok:
        free_b, total_b = torch.cuda.mem_get_info(0)
        vram_free  = round(free_b  / 1e9, 2)
        vram_total = round(total_b / 1e9, 2)
        vram_used  = round((total_b - free_b) / 1e9, 2)

    result = {
        "status":       "ok",
        "cuda":         cuda_ok,
        "gpu":          torch.cuda.get_device_name(0) if cuda_ok else "N/A",
        "vram_free_gb": vram_free,
        "vram_total_gb":vram_total,
        "vram_used_gb": vram_used,
        "model_path":   os.getenv("GGUF_MODEL_PATH", "not set"),
        "model_loaded": _llm is not None,
        "n_ctx":        int(os.getenv("N_CTX", "32768")),
    }
    log.info(f"[health] VRAM {vram_used}/{vram_total} GB  loaded={result['model_loaded']}")
    return result


# ── GET /info ──────────────────────────────────────────────────────────────────
@app.get(
    "/info",
    tags        = ["System"],
    summary     = "API Info",
    description = "Returns model config, supported categories, and pipeline metadata.",
)
def info():
    from app.categorizer import CATEGORY_RULES
    return {
        "version":             "2.0.0",
        "model":               "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF Q4_K_M",
        "model_path":          os.getenv("GGUF_MODEL_PATH", "not set"),
        "n_ctx":               int(os.getenv("N_CTX", "32768")),
        "n_gpu_layers":        int(os.getenv("N_GPU_LAYERS", "-1")),
        "max_new_tokens":      os.getenv("MAX_NEW_TOKENS", "0 (unlimited)"),
        "max_txn_chars":       int(os.getenv("MAX_TXN_CHARS", "30000")),
        "supported_categories":[rule[1] for rule in CATEGORY_RULES],
        "privacy":             "100% local — no data sent to external APIs",
        "pipeline_stages": [
            {"stage": 1, "name": "PDF Extraction",    "tool": "PyMuPDF",              "avg_time": "~0.1s"},
            {"stage": 2, "name": "Anonymisation",     "tool": "Regex (PAN/Mobile)",   "avg_time": "<1ms"},
            {"stage": 3, "name": "LLM Inference",     "tool": "Qwen3.5-27B GGUF",    "avg_time": "~15–40s"},
            {"stage": 4, "name": "Categorisation",    "tool": "Regex Rules (21 cats)","avg_time": "~1ms"},
        ],
    }