# Forschungsseminar

This repository contains two parts:

- `gam-lab/`: Next.js + React frontend for the GAM lab UI.
- `trainer-service/`: Python FastAPI service for dataset preprocessing and model training.

## Frontend (gam-lab)

```bash
cd gam-lab
npm install
npm run dev
```

Then open `http://localhost:3000`.

## Trainer service (trainer-service)

You need the datasets before running the frontend or backend:

- `trainer-service/download_datasets.py`: download datasets.
- `trainer-service/generate_models.py`: generate stored model outputs.

```bash
cd trainer-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn python_trainer:app --reload --port 4001
```

The API will be available at `http://localhost:4001` (or set `NEXT_PUBLIC_TRAINER_URL` / `TRAINER_URL` in `gam-lab` to match another port).

## Notes

- Frontend talks directly to the trainer service, so run both for full functionality.
