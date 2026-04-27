# Forschungsseminar

> **Repository visibility:** This repository should be **private** — it contains the MIMIC-IV derived dataset.

This repository contains two parts:

- `gam-lab/`: Next.js + React frontend for the GAM lab UI.
- `trainer-service/`: Python FastAPI service for dataset preprocessing and model training.

## Prerequisites

- **Node.js** ≥ 18.17 (check with `node -v`)
- **Python** ≥ 3.9 (check with `python3 --version`)
- **Git**

Before starting the frontend or backend, place the MIMIC-IV dataset file at `trainer-service/data/mimic4_mean_100_full.csv`.

## Frontend (gam-lab)

```bash
cd gam-lab
npm install
npm run dev

# For a production build: npm run build && npm start
```

Then open `http://localhost:3000`.


## Trainer service (trainer-service)

```bash
cd trainer-service
python3 -m venv .venv
source .venv/bin/activate
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Windows (cmd.exe): .venv\Scripts\activate.bat
pip install -r requirements.txt
#python -m trainer_service.datasets (for potentially downloading datasets)
#python -m trainer_service.generate_models (for generating preset models)
uvicorn trainer_service.api:app --reload --port 4001
```

The API will be available at `http://localhost:4001` (or set `NEXT_PUBLIC_TRAINER_URL` / `TRAINER_URL` in `gam-lab` to match another port).

Trainer saved models default to filesystem storage under `trainer-service/saved_models`.

