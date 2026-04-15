from __future__ import annotations

import json

from trainer_service.json_utils import to_jsonable
from trainer_service.paths import MODELS_DIR
from trainer_service.schemas import TrainRequest
from trainer_service.training import build_train_response


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    training_preset = {
        "model_type": "igann_interactive",
        "center_shapes": False,
        "seed": 3,
        "points": 250,
        "n_estimators": 100,
        "boost_rate": 0.1,
        "init_reg": 1,
        "elm_alpha": 1,
        "early_stopping": 50,
        "scale_y": True,
    }
    datasets = ["bike_hourly", "breast_cancer"]
    for dataset in datasets:
        request = TrainRequest(dataset=dataset, **training_preset)
        payload = to_jsonable(build_train_response(request))
        out_path = MODELS_DIR / f"{dataset}.json"
        with out_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
