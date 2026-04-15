# 🏦 Bank Statement AI — Automated Parser & Categorizer

> FastAPI + Qwen2.5-7B-INT4 powered bank statement extraction pipeline.  
> Extracts account details, transactions, and auto-categorizes spending from PDF statements.  
> Optimized for **RTX 4060 Ti 8GB VRAM** · Python 3.12 · CUDA 12.1 · Windows 10

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Hardware Requirements](#hardware-requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [API Reference](#api-reference)
- [Swagger UI](#swagger-ui)
- [Supported Banks](#supported-banks)
- [Category Rules](#category-rules)
- [Performance Benchmarks](#performance-benchmarks)
- [Troubleshooting](#troubleshooting)

---

## ✨ Features

- 📄 **PDF text extraction** via PyMuPDF (~0.1s per page — fastest available)
- 🤖 **LLM-powered parsing** using Qwen2.5-7B-Instruct-GPTQ-Int4 (~5GB VRAM)
- 🏷️ **Auto-categorization** of transactions via regex rules (20+ categories)
- ⚡ **Model warm-up at startup** — zero cold-start latency per request
- 🗂️ **Custom HuggingFace cache** — all models download to `H:\cache_hub\hub`
- 📊 **Transaction summary** — totals, net flow, per-category breakdown
- 🔍 **Swagger UI** at `/docs` and ReDoc at `/redoc`
- 🛡️ **Pydantic v2 schemas** — strict type validation on all inputs/outputs
- 🖼️ **Image fallback** — scanned/image PDFs handled via OCR flag

---

## 🏗️ Architecture

```
User Upload (PDF)
       │
       ▼
┌─────────────────┐
│  FastAPI /parse │  ← Receives multipart/form-data
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  extractor.py   │  ← PyMuPDF text + table extraction (~0.1s)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│           parser.py             │
│  ┌─────────────┐ ┌───────────┐  │
│  │parse_header │ │parse_txns │  │  ← Qwen2.5-7B INT4 on GPU
│  └─────────────┘ └───────────┘  │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│ categorizer.py  │  ← Regex rules, <1ms
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  JSON Response  │  ← account_details + transactions + summary + timings
└─────────────────┘
```

---

## 💻 Hardware Requirements

| Component | Minimum | Recommended (tested) |
|-----------|---------|----------------------|
| GPU | 8GB VRAM (NVIDIA) | RTX 4060 Ti 8GB |
| RAM | 12GB | 16GB |
| CPU | 4-core | i5 6-core |
| Storage (model) | 5GB free on model drive | H:\ drive |
| CUDA Version | 12.1 | 12.1 |
| Python | 3.10+ | 3.12 |
| OS | Windows 10/11, Linux | Windows 10 |

---

## 📁 Project Structure

```
bank-ai/
├── app/
│   ├── __init__.py          # Package marker (empty)
│   ├── main.py              # FastAPI app, lifespan, /parse endpoint
│   ├── extractor.py         # PyMuPDF PDF text + table extraction
│   ├── parser.py            # Qwen2.5 model loader + LLM inference
│   ├── categorizer.py       # Regex-based transaction categorizer
│   └── schemas.py           # Pydantic v2 models (request/response)
├── models/                  # Symlink → H:\cache_hub\hub
├── requirements.txt         # Python dependencies (excl. torch)
├── .env                     # Environment config (HF_HOME, MODEL_ID, etc.)
└── README.md
```

---

## 🚀 Installation

### Step 1 — Clone & Create Virtual Environment

```bash
git clone <your-repo-url> bank-ai
cd bank-ai
python -m venv .venv
.venv\Scripts\activate
```

### Step 2 — Install PyTorch (CUDA 12.1)

> ⚠️ Must be installed separately from the CUDA wheel index — NOT from PyPI.

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 ^
    --index-url https://download.pytorch.org/whl/cu121
```

### Step 3 — Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Create models/ Symlink

Run as **Administrator** in the project root:

```bash
mklink /D models "H:\cache_hub\hub"
```

### Step 5 — Create `.env`

```env
HF_HOME=H:/cache_hub/hub
TRANSFORMERS_CACHE=H:/cache_hub/hub
HF_HUB_CACHE=H:/cache_hub/hub
MODEL_ID=Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
DEVICE=cuda
MAX_NEW_TOKENS=2048
```

### Step 6 — Verify CUDA

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: 2.4.1+cu121  True  NVIDIA GeForce RTX 4060 Ti
```

---

## ⚙️ Configuration

All configuration lives in `.env`. Never commit this file.

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `H:/cache_hub/hub` | Root HuggingFace cache — models download here |
| `TRANSFORMERS_CACHE` | `H:/cache_hub/hub` | Transformers-specific cache path |
| `HF_HUB_CACHE` | `H:/cache_hub/hub` | HF Hub cache path |
| `MODEL_ID` | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` | HuggingFace model ID |
| `DEVICE` | `cuda` | Inference device (`cuda` or `cpu`) |
| `MAX_NEW_TOKENS` | `2048` | Max tokens to generate per LLM call |

---

## ▶️ Running the Server

```bash
# Development (with auto-reload)
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

# Production
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

> ⚠️ Always use `--workers 1` — the LLM is loaded into GPU memory once. Multiple workers = multiple model copies = OOM crash.

**First run:** The model (~4.5GB) will download to `H:\cache_hub\hub` automatically. Subsequent runs load from cache instantly.

### Startup Log (expected)

```
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
[STARTUP] Warming up LLM...
[INFO] Loading Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 from cache...
[INFO] Model loaded and ready.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 📡 API Reference

### `POST /parse`

Upload a bank statement PDF and receive structured JSON.

**Request**

```
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | `File` (PDF) | ✅ | Bank statement PDF (digital, not scanned) |

**Response `200 OK`**

```json
{
  "account_details": {
    "bank_name": "Axis Bank",
    "branch_name": "Tilak Road Branch, Rajahmundry",
    "bank_type": "Private",
    "account_number": "916010020446229",
    "account_holder": "Mahendra Kodavati",
    "address": "80-1-1/14 Flat No 101...",
    "pan": "AKOPK2643K",
    "mobile": "XXXXXX0257",
    "email": "koXXXX10@gmail.com",
    "ifsc_code": "UTIB0002978",
    "micr_code": "533211104",
    "account_type": "Prestige Savings Account",
    "statement_from": "2025-07-30",
    "statement_to": "2026-01-29",
    "opening_balance": "109429.47",
    "closing_balance": "0.72",
    "currency": "INR"
  },
  "transactions": [
    {
      "date": "2025-08-20",
      "value_date": "2025-08-20",
      "description": "ACH-DR-HDFC BANK LTD-431933083",
      "narration": "ACH-DR-HDFC BANK LTD",
      "cheque_no": "",
      "reference_no": "431933083",
      "transaction_type": "DEBIT",
      "debit": "29055.00",
      "credit": "",
      "balance": "80374.47",
      "branch_code": "2978",
      "remarks": "",
      "category": "Loan/EMI"
    }
  ],
  "summary": {
    "total_transactions": 35,
    "total_debit": 1142711.50,
    "total_credit": 1033282.75,
    "net_flow": -109428.75
  },
  "source_file": "Account_stmt_XX6229.pdf",
  "used_image_mode": false,
  "timings": {
    "pdf_extraction_s": 0.08,
    "llm_inference_s": 14.2,
    "total_s": 14.3
  }
}
```

**Error Responses**

| Code | Reason |
|------|--------|
| `400` | File is not a PDF |
| `422` | Validation error (Pydantic) |
| `500` | LLM inference failed / model not loaded |

---

### `GET /health`

Check server status, CUDA availability, and GPU memory.

**Response `200 OK`**

```json
{
  "status": "ok",
  "cuda": true,
  "gpu": "NVIDIA GeForce RTX 4060 Ti",
  "vram_free_gb": 3.12
}
```

---

### `GET /docs`

**Swagger UI** — interactive API documentation. Test endpoints directly from the browser.

```
http://localhost:   8000/docs
```

### `GET /redoc`

**ReDoc** — clean read-only API documentation.

```
http://localhost:8000/redoc
```

### `GET /openapi.json`

Raw OpenAPI 3.1 schema — use with Postman, Insomnia, or any API client.

```
http://localhost:8000/openapi.json
```

---

## 🔧 Enabling Rich Swagger Docs

Add the following to `main.py` to enhance Swagger with descriptions, tags, and examples:

```python
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI(
    title="Bank Statement AI",
    description="""
## 🏦 Bank Statement Parser API

Upload any Indian bank statement PDF and receive:
- ✅ Structured **account details**
- ✅ All **transactions** with dates, amounts, references
- ✅ Auto-assigned **spending categories**
- ✅ **Summary** with totals and net flow
- ✅ **Timing** breakdown for each pipeline stage

### Supported Banks
Axis Bank, HDFC Bank, SBI, ICICI Bank, Kotak, Yes Bank, IndusInd, PNB and more.

### Notes
- Only **digital PDFs** are supported (not scanned/image PDFs)
- Use `--workers 1` when running with uvicorn (GPU model is shared)
    """,
    version="1.0.0",
    contact={
        "name": "Rahul Hrushikesh",
        "email": "your@email.com",
    },
    license_info={
        "name": "MIT",
    },
    openapi_tags=[
        {"name": "parsing",  "description": "PDF upload and AI extraction endpoints"},
        {"name": "system",   "description": "Health check and server status"},
    ]
)
```

Then tag your endpoints:

```python
@app.post("/parse", tags=["parsing"], response_model=ParseResponse,
    summary="Parse Bank Statement PDF",
    description="Upload a bank statement PDF. Returns account details, all transactions with categories, and a financial summary.",
    responses={
        200: {"description": "Successfully parsed"},
        400: {"description": "Invalid file type"},
        500: {"description": "LLM inference error"},
    }
)
async def parse_statement(file: UploadFile = File(
    ...,
    description="Bank statement PDF file (digital, not scanned)"
)):
    ...


@app.get("/health", tags=["system"],
    summary="Server Health Check",
    description="Returns CUDA status, GPU name, and free VRAM."
)
def health():
    ...
```

---

## 🏛️ Supported Banks

| Bank | Statement Type | Notes |
|------|---------------|-------|
| Axis Bank | Digital PDF | ✅ Fully tested |
| HDFC Bank | Digital PDF | ✅ Supported |
| SBI | Digital PDF | ✅ Supported |
| ICICI Bank | Digital PDF | ✅ Supported |
| Kotak Mahindra | Digital PDF | ✅ Supported |
| Yes Bank | Digital PDF | ✅ Supported |
| IndusInd | Digital PDF | ✅ Supported |
| Scanned / Image PDFs | Any | ⚠️ Requires OCR fallback |

---

## 🏷️ Category Rules

Transactions are categorized using regex pattern matching in `categorizer.py`:

| Category | Matched Keywords |
|----------|-----------------|
| Income | salary, payroll, stipend, wages |
| Transfer | upi, imps, neft, rtgs, trf |
| Cash Withdrawal | atm, cash withdraw |
| Food & Dining | swiggy, zomato, restaurant, cafe, kfc, domino |
| Groceries | dmart, bigbazaar, grocer, supermarket |
| Utilities | electricity, jio, airtel, vodafone, apepdcl |
| Rent/Housing | rent, pg, hostel, lease, landlord |
| Transport | ola, uber, rapido, irctc, petrol, fastag |
| Shopping | amazon, flipkart, myntra, meesho |
| Entertainment | netflix, prime, hotstar, spotify, pvr |
| Healthcare | hospital, pharmacy, apollo, pharma |
| Education | school, college, cbse, udemy, fees |
| Insurance | lic, hdfc life, sbi life, premium |
| Loan/EMI | emi, loan, ach-dr, mortgage |
| Investment | zerodha, groww, sip, mutual fund |
| Bank Charges | charges, card charges, penalty |
| Interest Earned | interest, int.pd, savings interest |
| Tax | gst, tds, income tax |
| Travel | hotel, oyo, makemytrip, goibibo |
| Recharge | recharge, dth, tatasky |

---

## ⚡ Performance Benchmarks

Tested on RTX 4060 Ti 8GB · i5-12400F 6-core · 16GB RAM · Windows 10

| Stage | Time | Notes |
|-------|------|-------|
| PDF extraction (PyMuPDF) | ~0.08s | 3-page statement |
| LLM header parse | ~3–5s | First 3000 chars |
| LLM transaction parse | ~8–15s | Full statement text |
| Regex categorization | <1ms | All 35 transactions |
| **Total (typical)** | **~12–20s** | 3-page digital PDF |

**VRAM usage:**
- Qwen2.5-7B INT4 model: ~4.8GB
- Inference overhead: ~0.5GB
- Free headroom: ~2.7GB on 8GB card

---

## 🛠️ Troubleshooting

### `CUDA out of memory`
```bash
# Reduce max tokens in .env
MAX_NEW_TOKENS=1024

# Or force CPU (slow but works)
DEVICE=cpu
```

### `ModuleNotFoundError: No module named 'app'`
```bash
# Make sure __init__.py exists
type nul > app\__init__.py

# Run from project root, not from inside app/
cd bank-ai
uvicorn app.main:app --reload
```

### `Model downloads to C:\ instead of H:\`
```bash
# Confirm .env is loaded — add this debug line to main.py temporarily
import os; print(os.environ.get("HF_HOME"))
# Must print H:/cache_hub/hub before any import of transformers
```

### `torchaudio version not found`
```bash
# Install torch stack from CUDA wheel index only
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 ^
    --index-url https://download.pytorch.org/whl/cu121
```

### `JSON decode error from LLM response`
The LLM occasionally wraps JSON in markdown fences. The regex in `parser.py` handles this, but if it fails:
```python
# Add this fallback in parse_header() / parse_transactions()
raw = raw.replace("```json", "").replace("```", "").strip()
```

### `bitsandbytes` install fails on Windows
```bash
pip install bitsandbytes --prefer-binary
# If that fails, use the pre-built Windows wheel:
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```

---

## 📜 License

MIT License — free to use, modify, and distribute.

---

*Built with FastAPI · Qwen2.5 · PyMuPDF · HuggingFace Transformers*
