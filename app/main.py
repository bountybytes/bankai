import os
import tempfile
import time
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from app.extractor import extract_text
from app.parser import parse_header, parse_transactions, _load_model
from app.categorizer import categorize_transactions
from app.schemas import ParseResponse

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bank_ai.main")


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 60)
    log.info("STARTUP  — Bank Statement AI")
    log.info(f"MODEL_ID — {os.getenv('MODEL_ID', 'Qwen/Qwen2.5-7B-Instruct')}")
    log.info("Loading LLM into VRAM — may take 30–90s on first run...")
    _load_model()
    log.info("LLM ready ✓  Server is now accepting requests.")
    log.info("=" * 60)
    yield
    log.info("SHUTDOWN — releasing GPU resources...")


# ── App Init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bank Statement AI",
    description="""
## 🏦 Bank Statement Parser API

Automated AI pipeline that extracts structured data from Indian bank statement PDFs.

### What it does
Upload any digital bank statement PDF and receive:
- ✅ Structured **account details** (holder, account number, IFSC, PAN, etc.)
- ✅ All **transactions** with dates, amounts, cheque/reference numbers
- ✅ Auto-assigned **spending categories** (20+ categories via regex rules)
- ✅ **Financial summary** — total debits, credits, and net flow
- ✅ **Pipeline timings** — PDF extraction + LLM inference breakdown

### Supported Banks
Axis Bank, HDFC Bank, SBI, ICICI Bank, Kotak, Yes Bank, IndusInd, PNB and more.

### Tech Stack
- **PDF Extraction**: PyMuPDF (~0.1s per page)
- **LLM**: Qwen2.5-7B-Instruct NF4 (~5GB VRAM, ~45 t/s on RTX 4060 Ti)
- **Categorization**: Regex rules (<1ms)

### Notes
> ⚠️ Only **digital PDFs** are supported (not scanned/image-based PDFs).
> ⚠️ Run with `--workers 1` — GPU model is loaded once into VRAM.
    """,
    version="1.0.0",
    contact={"name": "Rahul Hrushikesh", "email": "your@email.com"},
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    openapi_tags=[
        {"name": "Parsing", "description": "Core PDF upload and AI extraction endpoints."},
        {"name": "System",  "description": "Health check, CUDA status, and server diagnostics."},
    ],
    lifespan=lifespan,
)


# ── /parse ────────────────────────────────────────────────────────────────────
@app.post(
    "/parse",
    tags=["Parsing"],
    response_model=ParseResponse,
    summary="Parse Bank Statement PDF",
    description="""
Upload a bank statement PDF file.

The pipeline runs in 3 stages:
1. **PDF Extraction** — PyMuPDF extracts raw text from all pages (~0.1s)
2. **LLM Inference** — Qwen2.5-7B parses account details and all transactions (~10–20s)
3. **Categorization** — Regex rules assign spending categories to each transaction (<1ms)
    """,
    responses={
        200: {"description": "Successfully parsed bank statement"},
        400: {"description": "Invalid file — only PDF files accepted",
              "content": {"application/json": {"example": {"detail": "Only PDF files supported"}}}},
        500: {"description": "LLM inference or extraction failed",
              "content": {"application/json": {"example": {"detail": "LLM inference error: <msg>"}}}},
    },
)
async def parse_statement(
    file: UploadFile = File(..., description="Bank statement PDF (digital only, not scanned)"),
):
    req_id = uuid.uuid4().hex[:8]  # short request ID for log correlation
    log.info(f"[{req_id}] ── NEW REQUEST ──────────────────────────────")
    log.info(f"[{req_id}] File     : {file.filename}  ({file.content_type})")

    if not file.filename.lower().endswith(".pdf"):
        log.warning(f"[{req_id}] Rejected — not a PDF: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    t0 = time.time()

    # ── Stage 1: PDF Extraction ───────────────────────────────────────────
    log.info(f"[{req_id}] Stage 1  : PDF extraction starting...")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        log.info(f"[{req_id}]           Saved to temp: {tmp_path}  ({len(content)/1024:.1f} KB)")
        raw_text = extract_text(tmp_path)
        os.unlink(tmp_path)
        log.info(f"[{req_id}]           Extracted {len(raw_text):,} chars from PDF")
    except Exception as e:
        log.error(f"[{req_id}] Stage 1 FAILED: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF extraction error: {str(e)}")

    t1 = time.time()
    log.info(f"[{req_id}]           ✓ Done in {t1 - t0:.3f}s")

    # ── Stage 2: LLM Parsing ──────────────────────────────────────────────
    log.info(f"[{req_id}] Stage 2  : LLM inference starting...")
    log.info(f"[{req_id}]           Parsing header (first 3000 chars)...")
    try:
        acc_details = parse_header(raw_text)
        log.info(f"[{req_id}]           Header parsed — account: {acc_details.get('account_number', 'N/A')}"
                 f"  holder: {acc_details.get('account_holder', 'N/A')}")

        log.info(f"[{req_id}]           Parsing transactions (full text: {len(raw_text):,} chars)...")
        transactions = parse_transactions(raw_text)
        log.info(f"[{req_id}]           Found {len(transactions)} transactions")
    except Exception as e:
        log.error(f"[{req_id}] Stage 2 FAILED: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"LLM inference error: {str(e)}")

    t2 = time.time()
    log.info(f"[{req_id}]           ✓ Done in {t2 - t1:.3f}s")

    # ── Stage 3: Categorization ───────────────────────────────────────────
    log.info(f"[{req_id}] Stage 3  : Categorizing {len(transactions)} transactions...")
    transactions = categorize_transactions(transactions)

    # Log category breakdown
    from collections import Counter
    cats = Counter(t.get("category", "Other") for t in transactions)
    for cat, count in cats.most_common():
        log.info(f"[{req_id}]             {cat:<25} {count} txn(s)")

    t3 = time.time()
    log.info(f"[{req_id}]           ✓ Done in {(t3 - t2) * 1000:.2f}ms")

    # ── Summary ───────────────────────────────────────────────────────────
    debits  = []
    credits = []
    for t in transactions:
        try:
            if t.get("debit")  and str(t["debit"]).strip():
                debits.append(float(t["debit"]))
            if t.get("credit") and str(t["credit"]).strip():
                credits.append(float(t["credit"]))
        except (ValueError, TypeError) as e:
            log.warning(f"[{req_id}] Could not parse amount in txn: {t} — {e}")

    summary = {
        "total_transactions": len(transactions),
        "total_debit":        round(sum(debits), 2),
        "total_credit":       round(sum(credits), 2),
        "net_flow":           round(sum(credits) - sum(debits), 2),
    }

    total_s = round(time.time() - t0, 3)
    log.info(f"[{req_id}] ── COMPLETE ─────────────────────────────────")
    log.info(f"[{req_id}] Transactions : {summary['total_transactions']}")
    log.info(f"[{req_id}] Total Debit  : ₹{summary['total_debit']:,.2f}")
    log.info(f"[{req_id}] Total Credit : ₹{summary['total_credit']:,.2f}")
    log.info(f"[{req_id}] Net Flow     : ₹{summary['net_flow']:,.2f}")
    log.info(f"[{req_id}] Total Time   : {total_s}s  "
             f"(pdf={round(t1-t0,3)}s  llm={round(t2-t1,3)}s  cat={round((t3-t2)*1000,2)}ms)")

    return JSONResponse(content={
        "account_details": acc_details,
        "transactions":    transactions,
        "summary":         summary,
        "source_file":     file.filename,
        "used_image_mode": False,
        "timings": {
            "pdf_extraction_s": round(t1 - t0, 3),
            "llm_inference_s":  round(t2 - t1, 3),
            "total_s":          total_s,
        },
    })


# ── /health ───────────────────────────────────────────────────────────────────
@app.get(
    "/health",
    tags=["System"],
    summary="Server Health Check",
    description="Returns CUDA availability, GPU name, free VRAM, and model load status.",
)
def health():
    import torch
    from app.parser import _model, _tokenizer

    cuda_available = torch.cuda.is_available()
    vram_free = vram_total = vram_used = 0.0
    if cuda_available:
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        vram_free  = round(free_bytes  / 1e9, 2)
        vram_total = round(total_bytes / 1e9, 2)
        vram_used  = round((total_bytes - free_bytes) / 1e9, 2)

    result = {
        "status":        "ok",
        "cuda":          cuda_available,
        "gpu":           torch.cuda.get_device_name(0) if cuda_available else "N/A",
        "vram_free_gb":  vram_free,
        "vram_total_gb": vram_total,
        "vram_used_gb":  vram_used,
        "model_id":      os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct"),
        "model_loaded":  _model is not None and _tokenizer is not None,
        "device":        os.getenv("DEVICE", "cuda"),
    }
    log.info(f"[health] VRAM {vram_used}/{vram_total} GB  model_loaded={result['model_loaded']}")
    return result


# ── /info ─────────────────────────────────────────────────────────────────────
@app.get(
    "/info",
    tags=["System"],
    summary="API & Category Info",
    description="Returns supported categories, model config, and pipeline metadata.",
)
def info():
    from app.categorizer import CATEGORY_RULES
    return {
        "version":        "1.0.0",
        "model_id":       os.getenv("MODEL_ID"),
        "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", 2048)),
        "hf_cache":       os.getenv("HF_HOME"),
        "supported_categories": [rule[1] for rule in CATEGORY_RULES],
        "pipeline_stages": [
            {"stage": 1, "name": "PDF Extraction",    "tool": "PyMuPDF",          "avg_time": "~0.1s"},
            {"stage": 2, "name": "Header Parsing",    "tool": "Qwen2.5-7B-NF4",   "avg_time": "~3-5s"},
            {"stage": 3, "name": "Transaction Parse", "tool": "Qwen2.5-7B-NF4",   "avg_time": "~8-15s"},
            {"stage": 4, "name": "Categorization",    "tool": "Regex Rules",       "avg_time": "<1ms"},
        ],
    }