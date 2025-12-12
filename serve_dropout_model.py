"""FastAPI service exposing the dropout prediction pipeline.

Run locally for development:
    uvicorn serve_dropout_model:app --reload --port 8001
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
MODEL_PATH = Path("artifacts/dropout_early21.joblib")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Missing trained pipeline artifact. Execute main.ipynb to produce artifacts/dropout_early21.joblib first."
    )

artifact = joblib.load(MODEL_PATH)
PIPELINE = artifact["pipeline"]
METADATA = artifact["metadata"]
NUMERIC_FEATURES: List[str] = METADATA["numeric_features"]
CATEGORICAL_FEATURES: List[str] = METADATA["categorical_features"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class StudentFeatures(BaseModel):
    """Feature schema produced by the notebook aggregation step."""

    id_student: int = Field(..., description="Primary key as provided by the OULAD dataset")
    total_clicks: float = 0
    active_days: float = 0
    avg_clicks_per_day: float = 0
    avg_clicks_per_active_day: float = 0
    clicks_0_7: float = 0
    clicks_8_14: float = 0
    trend_click: float = 0
    ratio_click: float = 0
    clicks_15_21: float = 0
    num_assessments: float = 0
    avg_score: float = 0
    max_score: float = 0
    min_score: float = 0
    score_std: float = 0
    last_score: float = 0
    pass_rate: float = 0
    reg_day: float = 0
    registered_before_start: float = 0
    days_since_last_login: float = 0
    inactivity_streak: float = 0
    gender: str = Field(..., description="Use same categories as training data")
    age_band: str = Field(..., description="Use the same buckets as the training data")

    def to_row(self) -> dict:
        row = {feature: getattr(self, feature) for feature in ALL_FEATURES}
        row["id_student"] = self.id_student
        return row


class PredictRequest(BaseModel):
    students: List[StudentFeatures]
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Decision threshold for dropout class")


class PredictResponseItem(BaseModel):
    id_student: int
    dropout_probability: float
    dropout_prediction: int


class PredictResponse(BaseModel):
    model_name: str
    module: str
    presentation: str
    early_days: int
    results: List[PredictResponseItem]


app = FastAPI(title="Dropout Predictor API", version="1.0.0")


def _score_dataframe(df: pd.DataFrame, threshold: float) -> List[PredictResponseItem]:
    if df.empty:
        return []
    features = df[ALL_FEATURES]
    probabilities = PIPELINE.predict_proba(features)[:, 1]
    labels = (probabilities >= threshold).astype(int)
    response: List[PredictResponseItem] = []
    for student_id, prob, label in zip(df["id_student"], probabilities, labels, strict=False):
        response.append(
            PredictResponseItem(
                id_student=int(student_id),
                dropout_probability=float(prob),
                dropout_prediction=int(label),
            )
        )
    return response


@app.get("/health")
def healthcheck() -> dict:
    return {
        "status": "ok",
        "model_name": METADATA["model_name"],
        "module": METADATA["module"],
        "presentation": METADATA["presentation"],
        "early_days": METADATA["early_days"],
        "features": ALL_FEATURES,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if not payload.students:
        raise HTTPException(status_code=400, detail="Payload must include at least one student record")

    rows = [student.to_row() for student in payload.students]
    frame = pd.DataFrame(rows)
    missing = set(ALL_FEATURES) - set(frame.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required features: {sorted(missing)}")

    results = _score_dataframe(frame, threshold=payload.threshold)

    return PredictResponse(
        model_name=METADATA["model_name"],
        module=METADATA["module"],
        presentation=METADATA["presentation"],
        early_days=METADATA["early_days"],
        results=results,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve_dropout_model:app", host="0.0.0.0", port=8001, reload=True)
