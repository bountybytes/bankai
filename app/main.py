"""
main.py — FastAPI Bank Statement AI
Async job polling pattern to bypass RunPod/Cloudflare 100s timeout
"""

import os, tempfile, time, logging, uuid, asyncio
from collections import Counter
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from app.extractor   import extract_text, anonymize_sensitive, restore_sensitive
from app.parser      import parse_header, parse_transactions, load_model
from app.categorizer import categorize_transactions
from app.schemas     import ParseResponse

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("bank_ai.main")

# ── In-memory job store ────────────────────────────────────────────────────────
# { job_id: { "status": "processing"|"done"|"error", "result": {...}, "error": "..." } }
_jobs: dict = {}
_executor   = ThreadPoolExecutor(max_workers=1)  # 1 GPU — serialize inference


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 60)
    log.info("STARTUP  — Bank Statement AI (GGUF)")
    log.info(f"  Model  : {os.getenv('MODEL_PATH', 'not set')}")
    log.info(f"  n_ctx  : {os.getenv('N_CTX', '65536')}")
    log.info("Loading Qwen2.5-Coder-14B GGUF (~9 GB) ...")
    load_model()
    log.info("Model ready ✓  Server is now accepting requests.")
    log.info("=" * 60)
    yield
    log.info("SHUTDOWN — releasing resources.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Bank Statement AI",
    description = """
## Automated Bank Statement Parser — Qwen2.5-Coder-14B GGUF

Upload any digital Indian bank statement PDF and receive structured JSON.

### ⚡ Use Async Endpoints to avoid timeouts
- `POST /parse/submit` — submit job, returns `job_id` instantly
- `GET  /parse/result/{job_id}` — poll until `status: done`
- `POST /parse` — sync (only for < 60s statements, may 524 timeout)
""",
    version     = "3.0.0",
    lifespan    = lifespan,
)


# ── Core pipeline (runs in thread) ────────────────────────────────────────────
def _run_pipeline(pdf_bytes: bytes, filename: str) -> dict:
    req_id = uuid.uuid4().hex[:8]
    t0 = time.time()

    # ── Save PDF — keep alive for GLM-OCR page rendering ──
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(pdf_bytes)
    tmp.flush()
    tmp_path = tmp.name
    tmp.close()                          # close handle but DON'T delete yet

    raw_text = extract_text(tmp_path)
    t1 = time.time()

    anon_text, restore_map = anonymize_sensitive(raw_text)

    # Header — Qwen (text only, ~512 tokens out)
    acc_details = parse_header(anon_text)
    acc_details = restore_sensitive(acc_details, restore_map)

    # Transactions — GLM-OCR (pass pdf_path so it can render pages)
    transactions = parse_transactions(anon_text, pdf_path=tmp_path)
    transactions = [restore_sensitive(t, restore_map) for t in transactions]
    t2 = time.time()

    os.unlink(tmp_path)                  # ← delete AFTER inference completes

    # Categorise
    transactions = categorize_transactions(transactions)
    cats = Counter(t.get("category", "Other") for t in transactions)
    t3 = time.time()

    # Summary
    debits, credits = [], []
    for txn in transactions:
        try:
            if txn.get("debit")  and str(txn["debit"]).strip():  debits.append(float(txn["debit"]))
            if txn.get("credit") and str(txn["credit"]).strip(): credits.append(float(txn["credit"]))
        except (ValueError, TypeError):
            pass

    total_s = round(time.time() - t0, 3)
    log.info(f"[{req_id}] COMPLETE — {len(transactions)} txns  {total_s}s total")

    return {
        "account_details": acc_details,
        "transactions":    transactions,
        "summary": {
            "total_transactions": len(transactions),
            "total_debit":        round(sum(debits), 2),
            "total_credit":       round(sum(credits), 2),
            "net_flow":           round(sum(credits) - sum(debits), 2),
        },
        "source_file":      filename,
        "used_image_mode":  False,
        "timings": {
            "pdf_extraction_s":  round(t1 - t0, 3),
            "llm_inference_s":   round(t2 - t1, 3),
            "categorization_ms": round((t3 - t2) * 1000, 2),
            "total_s":           total_s,
        },
    }


# ── POST /parse/submit  (RECOMMENDED — async, no timeout) ─────────────────────
@app.post("/parse/submit", tags=["Parsing"],
    summary="Submit PDF for async parsing (no timeout)",
    description="Returns a `job_id` immediately. Poll `GET /parse/result/{job_id}` until `status` is `done`.")
async def parse_submit(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Bank statement PDF"),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    job_id    = uuid.uuid4().hex[:12]
    pdf_bytes = await file.read()
    filename  = file.filename

    _jobs[job_id] = {"status": "processing", "submitted_at": time.time()}
    log.info(f"[job:{job_id}] Submitted — {filename}  ({len(pdf_bytes)/1024:.1f} KB)")

    # Run in background thread (non-blocking for uvicorn)
    loop = asyncio.get_event_loop()
    async def _bg():
        try:
            result = await loop.run_in_executor(_executor, _run_pipeline, pdf_bytes, filename)
            _jobs[job_id] = {"status": "done", "result": result, "completed_at": time.time()}
        except Exception as e:
            log.error(f"[job:{job_id}] FAILED: {e}", exc_info=True)
            _jobs[job_id] = {"status": "error", "error": str(e)}

    background_tasks.add_task(_bg)

    return {
        "job_id":    job_id,
        "status":    "processing",
        "poll_url":  f"/parse/result/{job_id}",
        "message":   "Poll /parse/result/{job_id} every 5 seconds until status is 'done'",
    }


# ── GET /parse/result/{job_id}  ────────────────────────────────────────────────
@app.get("/parse/result/{job_id}", tags=["Parsing"],
    summary="Poll job result",
    description="Returns `status: processing` while running, `status: done` with full result when complete.")
def parse_result(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job = _jobs[job_id]

    if job["status"] == "processing":
        elapsed = round(time.time() - job["submitted_at"], 1)
        return {"status": "processing", "elapsed_s": elapsed, "job_id": job_id}

    if job["status"] == "error":
        return JSONResponse(status_code=500, content={"status": "error", "error": job["error"]})

    return {"status": "done", "job_id": job_id, **job["result"]}


# ── GET /parse/jobs  ───────────────────────────────────────────────────────────
@app.get("/parse/jobs", tags=["Parsing"], summary="List all jobs")
def list_jobs():
    return {
        jid: {"status": j["status"], "elapsed_s": round(time.time() - j.get("submitted_at", time.time()), 1)}
        for jid, j in _jobs.items()
    }


# ── POST /parse  (sync — kept for compatibility, may 524 on large PDFs) ────────
@app.post("/parse", tags=["Parsing"],
    summary="Sync parse (use /parse/submit for large PDFs)",
    description="⚠️ May 524 timeout via RunPod proxy for large statements. Use `/parse/submit` instead.")
async def parse_sync(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    pdf_bytes = await file.read()
    loop      = asyncio.get_event_loop()

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, _run_pipeline, pdf_bytes, file.filename),
            timeout=90.0,
        )
        return JSONResponse(content=result)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="LLM inference exceeded 90s. Use POST /parse/submit + GET /parse/result/{job_id} instead."
        )


# ── GET /health ────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"], summary="Health Check")
def health():
    import torch
    from app.parser import _llm

    cuda_ok = torch.cuda.is_available()
    vram_free = vram_total = vram_used = 0.0
    if cuda_ok:
        free_b, total_b = torch.cuda.mem_get_info(0)
        vram_free  = round(free_b  / 1e9, 2)
        vram_total = round(total_b / 1e9, 2)
        vram_used  = round((total_b - free_b) / 1e9, 2)

    jobs_summary = Counter(j["status"] for j in _jobs.values())
    log.info(f"[health] VRAM {vram_used}/{vram_total} GB  loaded={_llm is not None}")

    return {
        "status":        "ok",
        "model_loaded":  _llm is not None,
        "cuda":          cuda_ok,
        "gpu":           torch.cuda.get_device_name(0) if cuda_ok else "N/A",
        "vram_used_gb":  vram_used,
        "vram_total_gb": vram_total,
        "vram_free_gb":  vram_free,
        "jobs":          dict(jobs_summary),
        "model_path":    os.getenv("MODEL_PATH", "not set"),
        "n_ctx":         int(os.getenv("N_CTX", "65536")),
    }


# ── GET /info ──────────────────────────────────────────────────────────────────
@app.get("/info", tags=["System"], summary="API Info")
def info():
    return {
        "version":        "3.0.0",
        "model":          "Qwen2.5-Coder-14B-Instruct Q4_K_M GGUF",
        "async_endpoint": "POST /parse/submit  →  GET /parse/result/{job_id}",
        "sync_endpoint":  "POST /parse  (90s timeout — may fail on large PDFs)",
        "n_ctx":          int(os.getenv("N_CTX", "65536")),
        "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "16384")),
    }
