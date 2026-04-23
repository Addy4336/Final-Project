# VisionQuery: OCR + VQA + Satellite Multimodal System

This repository is a FastAPI + React multimodal system that supports:

- OCR-driven document question answering
- Visual Question Answering (ViLT base and fine-tuned)
- Satellite scene analysis with visual grounding
- Optional YOLO object overlay for generic VQA prompts

## Research Gap and Objective

Do not change this framing (as requested):

Existing methods fail to integrate text inside images.
Attention models focus visually but ignore semantic text.
Fusion methods lack contextual alignment.
Remote sensing adds complexity (scale, noise).

Gap: No unified OCR + VQA reasoning framework.

Objective: OCR + VQA Hybrid Model

- Extracts textual + visual features simultaneously
- Enables contextual reasoning
- Improves semantic understanding
- Designed for real-world complexity

## Current Architecture

- Backend: `backend/app.py` with mode routing (`AUTO`, `OCR`, `SATELLITE`, `VQA`)
- OCR: `backend/models/ocr.py` hybrid EasyOCR + Tesseract pipeline
- VQA: `backend/models/vqa.py` ViLT wrapper with base/fine-tuned behaviors
- Satellite: `backend/models/satellite.py` HSV-based analysis and overlays
- Frontend (legacy): `frontend/index.html` + `frontend/components/VisionQueryApp_v2.jsx`
- Frontend (recommended): `frontend-next` (Next.js + TypeScript + Tailwind + Framer Motion)

## Setup (Windows)

Recommended Python version: 3.11 (3.10+ supported).

1. Create and activate a virtual environment.

```powershell
cd "d:\Projects\Final Project"
python -m venv .venv_local
.\.venv_local\Scripts\Activate.ps1
```

2. Install Python dependencies.

```powershell
pip install -r requirements.txt
```

Optional one-command setup:

```powershell
cd "d:\Projects\Final Project"
.\setup_windows.ps1
```

3. Install Tesseract OCR (required for best OCR quality).

- Install from official Windows installer.
- Default path expected by backend:
  - `C:\Program Files\Tesseract-OCR\tesseract.exe`
  - `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`

If installed elsewhere, add Tesseract to system `PATH`.

## Run the App

```powershell
cd "d:\Projects\Final Project"
.\.venv_local\Scripts\Activate.ps1
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Open in browser:

- `http://127.0.0.1:8000`

## Run the Professional Next.js UI (Recommended)

Use two terminals.

Terminal 1 (backend):

```powershell
cd "d:\Projects\Final Project"
.\.venv_local\Scripts\Activate.ps1
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Terminal 2 (Next.js UI):

```powershell
cd "d:\Projects\Final Project\frontend-next"
Copy-Item .env.local.example .env.local
npm install
npm run dev
```

Open in browser:

- `http://localhost:3000`

Notes:

- The UI calls FastAPI directly using `NEXT_PUBLIC_BACKEND_URL` (default: `http://127.0.0.1:8000`) to avoid dev-proxy timeout on long OCR requests.
- First OCR/VQA request can be slow on CPU due to model warmup.
- Subsequent requests are typically faster.

Production build check:

```powershell
cd "d:\Projects\Final Project\frontend-next"
npm run build
```

Run production server:

```powershell
npm start
```

## Useful Health Checks

General health:

```powershell
curl http://127.0.0.1:8000/api/health
```

OCR engine diagnostics:

```powershell
curl http://127.0.0.1:8000/api/ocr/health
```

The OCR health endpoint reports EasyOCR and Tesseract readiness.

Next UI to backend sanity check:

```powershell
curl http://127.0.0.1:8000/api/health
curl http://127.0.0.1:3000
```

## API Notes

Main endpoint:

- `POST /api/vqa`

Key multipart fields:

- `image`
- `question`
- `mode` (`AUTO`, `OCR`, `SATELLITE`, `VQA`)
- `analysis_mode` (legacy alias, still supported)
- `brightness`, `contrast`, and optional `region_*`

## Known Constraints

- First model load can be slow (transformers + torch warmup).
- OCR quality depends heavily on image clarity and Tesseract presence.
- Satellite analysis is rule-based and should be treated as an interpretable baseline.

## Academic Usage

This codebase is a development-stage research prototype intended for academic demonstration and iterative experimental refinement.
