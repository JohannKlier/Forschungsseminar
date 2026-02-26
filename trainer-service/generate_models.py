import json
from pathlib import Path

from python_trainer import TrainRequest, build_train_response, MODELS_DIR, _to_jsonable


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    defaults = {
        "seed": 3,
        "points": 250,
    }
    datasets = ["bike_hourly", "breast_cancer"]
    for dataset in datasets:
        request = TrainRequest(dataset=dataset, **defaults)
        payload = _to_jsonable(build_train_response(request))
        out_path = MODELS_DIR / f"{dataset}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
