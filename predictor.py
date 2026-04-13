from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

NORMAL_RANGES: dict[str, tuple[float, float]] = {
    "WBC": (4000, 10000),
    "LY%": (20.0, 40.0),
    "MO%": (2.0, 10.0),
    "NE%": (40.0, 80.0),
    "EO%": (1.0, 6.0),
    "BA%": (0.0, 1.0),
    "LY#": (1.0, 3.0),
    "MO#": (0.2, 1.0),
    "NE#": (2.0, 7.0),
    "EO#": (0.02, 0.5),
    "BA#": (0.02, 0.1),
    "RBC": (3.8, 4.8),
    "HGB": (12.0, 15.0),
    "HCT": (36.0, 46.0),
    "MCV": (83.0, 101.0),
    "MCHC": (31.5, 34.5),
    "MCH": (27.0, 32.0),
    "RDW": (11.6, 14.0),
    "PLT": (150.0, 410.0),
    "MPV": (5.4, 10.2),
}

DISEASE_RULES: dict[str, list[str]] = {
    "Infection": ["WBC"],
    "AllergicReaction": ["EO%", "EO#"],
    "Anemia": ["HGB"],
    "ClottingIssue": ["PLT"],
}


class CBCPredictor:
    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        model_dir = project_root / "modeldir"

        model_path = model_dir / "xgb_disease_model_v1.pkl"
        label_binarizer_path = model_dir / "label_binarizer.pkl"
        feature_columns_path = model_dir / "feature_columns.pkl"
        defaults_csv_path = project_root / "cbc_dataframe_with_disease.csv"

        self.model = joblib.load(model_path)
        self.mlb = joblib.load(label_binarizer_path)
        self.feature_columns: list[str] = joblib.load(feature_columns_path)
        self.label_names = list(self.mlb.classes_)
        self.raw_feature_defaults = self._build_raw_defaults(defaults_csv_path)

    def _build_raw_defaults(self, defaults_csv_path: Path) -> dict[str, float]:
        if defaults_csv_path.exists():
            df = pd.read_csv(defaults_csv_path)
            medians = df.apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
            defaults: dict[str, float] = {}
            for feat in NORMAL_RANGES:
                if feat in medians.index and pd.notna(medians[feat]):
                    defaults[feat] = float(medians[feat])
            if defaults:
                return defaults

        return {feat: float((low + high) / 2.0) for feat, (low, high) in NORMAL_RANGES.items()}

    def _build_feature_row(self, raw_values: dict[str, Any]) -> pd.DataFrame:
        row = pd.DataFrame([raw_values]).apply(pd.to_numeric, errors="coerce")

        for feat in NORMAL_RANGES:
            if feat not in row.columns or pd.isna(row.at[0, feat]):
                row[feat] = self.raw_feature_defaults.get(feat, 0.0)

        for feat, (low, high) in NORMAL_RANGES.items():
            val = float(row.at[0, feat])
            row[f"{feat}_OOR"] = int((val < low) or (val > high))
            row[f"{feat}_low"] = int(val < low)
            row[f"{feat}_high"] = int(val > high)
            row[f"{feat}_dev"] = max(0.0, low - val, val - high)

        for disease, triggers in DISEASE_RULES.items():
            flag_cols = [f"{f}_OOR" for f in triggers if f"{f}_OOR" in row.columns]
            row[f"rule_{disease}"] = float(sum(row.at[0, c] for c in flag_cols)) if flag_cols else 0.0

        for col in self.feature_columns:
            if col not in row.columns:
                row[col] = 0.0

        return row[self.feature_columns]

    def predict_patient(self, raw_values: dict[str, Any]) -> pd.DataFrame:
        feature_row = self._build_feature_row(raw_values)
        probs = np.array([est.predict_proba(feature_row)[0][1] for est in self.model.estimators_])
        preds = (probs >= 0.5).astype(int)

        return pd.DataFrame(
            {
                "Disease": self.label_names,
                "Predicted": preds.astype(bool),
                "Probability": (probs.round(6) * 100),
            }
        ).sort_values("Probability", ascending=False, ignore_index=True)


_predictor_singleton: CBCPredictor | None = None


def get_predictor() -> CBCPredictor:
    global _predictor_singleton
    if _predictor_singleton is None:
        _predictor_singleton = CBCPredictor()
    return _predictor_singleton


def predict_patient(raw_values: dict[str, Any]) -> pd.DataFrame:
    return get_predictor().predict_patient(raw_values)
