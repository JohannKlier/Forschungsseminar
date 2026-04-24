# Forschungsseminar

> **Repository visibility:** This repository should be **private** — it contains the MIMIC-IV derived dataset.

This repository contains two parts:

- `gam-lab/`: Next.js + React frontend for the GAM lab UI.
- `trainer-service/`: Python FastAPI service for dataset preprocessing and model training.

## Prerequisites

- **Node.js** ≥ 18.17 (check with `node -v`)
- **Python** ≥ 3.9 (check with `python3 --version`)
- **Git** — required for installing the `igann` package from GitHub (`pip install` fetches it via git)

## Frontend (gam-lab)

```bash
cd gam-lab
npm install
npm run dev
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
python -m trainer_service.datasets
python -m trainer_service.generate_models
uvicorn trainer_service.api:app --reload --port 4001
```

The API will be available at `http://localhost:4001` (or set `NEXT_PUBLIC_TRAINER_URL` / `TRAINER_URL` in `gam-lab` to match another port).
The MIMIC-IV option expects `mimic4_mean_100_full.csv` at `trainer-service/data/mimic4_mean_100_full.csv`.
This file is checked into the repository (repo must be private) — no separate download needed.

Trainer saved models default to filesystem storage under `trainer-service/saved_models`.
For deployment, switch them to Postgres:

```bash
SAVED_MODELS_STORAGE=postgres
SAVED_MODELS_DATABASE_URL=postgres://...
# Set when your managed Postgres provider requires TLS.
SAVED_MODELS_DATABASE_SSL=require
```

The trainer creates the saved-model table automatically on first use.

### Deployment notes

- Treat `trainer-service/data` and `trainer-service/models` as build artifacts and bake them into the image.
- Treat `trainer-service/saved_models` as local-development fallback storage only.
- Prefer platform-managed logs from `stdout`/`stderr` over application log files.
