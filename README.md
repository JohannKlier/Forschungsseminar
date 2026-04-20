# Forschungsseminar

This repository contains two parts:

- `gam-lab/`: Next.js + React frontend for the GAM lab UI.
- `trainer-service/`: Python FastAPI service for dataset preprocessing and model training.

## Run Everything In Development

From the repository root:

```bash
./dev.sh
```

This starts the trainer service on `http://localhost:4001` and the frontend on `http://localhost:3000`.

## Frontend (gam-lab)

```bash
cd gam-lab
npm install
npm run dev
```

Then open `http://localhost:3000`.

### Frontend environment

For local development the audit log defaults to filesystem storage under `gam-lab/data/audit`.
For deployment, switch the audit log to Postgres:

```bash
AUDIT_STORAGE=postgres
DATABASE_URL=postgres://...
# Set when your managed Postgres provider requires TLS.
AUDIT_DATABASE_SSL=require
TRAINER_URL=http://trainer-service:4001
NEXT_PUBLIC_TRAINER_URL=http://trainer-service:4001
```

The audit schema is created automatically by the Next.js server on first use.
Raw form values are no longer logged unless an element explicitly opts in with `data-audit-value="allow"`.

## Trainer service (trainer-service)

You need the datasets before running the frontend or backend:

- Core package code now lives under `trainer-service/trainer_service/`:
  - `api.py`: FastAPI routes and app wiring.
  - `training.py`: training orchestration and shape logic.
  - `preprocessing/`: dataset-specific preprocessing modules.
  - `datasets.py`: dataset download helpers.
  - `generate_models.py`: preset model generation.
  - `storage.py` and `model_store.py`: saved-model and preset-model persistence.

```bash
cd trainer-service
python -m venv .venv
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

## Notes

- Frontend talks directly to the trainer service, so run both for full functionality.
