from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel


class SaveModelRequest(BaseModel):
    name: str
    payload: Dict


class TrainRequest(BaseModel):
    dataset: str
    model_type: str = "igann_interactive"
    center_shapes: bool = False
    selected_features: List[str] = []
    selected_interactions: List[str] | None = None
    selected_operations: List[Dict] = []
    seed: int = 3
    points: int | None = 250
    n_estimators: int = 100
    boost_rate: float = 0.1
    init_reg: float = 1
    elm_alpha: float = 1
    early_stopping: int = 50
    scale_y: bool = True


class RefitRequest(TrainRequest):
    partials: List[Dict]
    locked_features: List[str] = []
    feature_modes: Dict[str, str] = {}
