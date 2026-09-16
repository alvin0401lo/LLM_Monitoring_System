from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODEL_PATH = ROOT / "model" / "failure_model.pkl"
FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"


def create_demo_dataset(rows: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    air = rng.normal(300.0, 2.0, rows)
    process = air + rng.normal(10.0, 1.2, rows)
    rpm = rng.normal(1500.0, 180.0, rows).clip(900, 2200)
    torque = rng.normal(42.0, 10.0, rows).clip(15, 75)
    wear = rng.uniform(0, 250, rows)
    risk_score = (
        0.028 * (wear - 120)
        + 0.07 * (torque - 42)
        + 0.0025 * (process - 310) ** 2
        + 0.004 * np.abs(rpm - 1500)
        - 2.0
    )
    probability = 1 / (1 + np.exp(-risk_score))
    failure = (rng.random(rows) < probability).astype(int)
    return pd.DataFrame(
        {
            "UDI": np.arange(1, rows + 1),
            "Product ID": [f"M-{index:04d}" for index in range(1, rows + 1)],
            "Type": rng.choice(["L", "M", "H"], rows, p=[0.5, 0.3, 0.2]),
            FEATURES[0]: air.round(2),
            FEATURES[1]: process.round(2),
            FEATURES[2]: rpm.round(2),
            FEATURES[3]: torque.round(2),
            FEATURES[4]: wear.round(2),
            TARGET: failure,
        }
    )


def ensure_dataset() -> pd.DataFrame:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    dataset = create_demo_dataset()
    dataset.to_csv(DATA_PATH, index=False)
    return dataset


def train_model() -> dict[str, float]:
    dataset = ensure_dataset()
    missing = [column for column in FEATURES + [TARGET] if column not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")

    X = dataset[FEATURES]
    y = dataset[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=180, random_state=42, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "feature_medians": X.median().to_dict(),
            "data_signature": (DATA_PATH.stat().st_size, DATA_PATH.stat().st_mtime_ns),
        },
        MODEL_PATH,
    )
    return {"accuracy": float(model.score(X_test, y_test)), "rows": float(len(dataset))}


if __name__ == "__main__":
    metrics = train_model()
    print(f"Trained on {int(metrics['rows'])} rows; holdout accuracy: {metrics['accuracy']:.1%}")
