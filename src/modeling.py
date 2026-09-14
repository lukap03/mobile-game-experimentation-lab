"""Leakage-safe D7 churn modelling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "churned_d7"
CATEGORICAL_FEATURES = ["variant", "platform", "country", "acquisition_channel", "age_band"]
NUMERIC_FEATURES = ["tutorial_completed", "sessions_d1", "playtime_minutes_d1", "levels_completed_d1", "revenue_d1"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
LEAKAGE_COLUMNS = {"retained_d7", "churned_d7", "revenue_d7"}


def build_pipeline() -> Pipeline:
    """Build preprocessing and logistic regression as one fitted artifact."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessing = ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)])
    return Pipeline([("preprocessing", preprocessing), ("classifier", LogisticRegression(max_iter=1_000, class_weight="balanced"))])


def train_and_evaluate(data: pd.DataFrame, *, seed: int = 42) -> tuple[Pipeline, dict[str, object], pd.DataFrame]:
    """Fit on day-1 features and evaluate on a stratified holdout."""
    if LEAKAGE_COLUMNS.intersection(FEATURES):
        raise RuntimeError("Outcome leakage detected in model features")
    train, test = train_test_split(data, test_size=0.25, random_state=seed, stratify=data[TARGET])
    model = build_pipeline().fit(train[FEATURES], train[TARGET])
    probability = model.predict_proba(test[FEATURES])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    matrix = confusion_matrix(test[TARGET], prediction)
    metrics: dict[str, object] = {
        "roc_auc": float(roc_auc_score(test[TARGET], probability)),
        "pr_auc": float(average_precision_score(test[TARGET], probability)),
        "precision": float(precision_score(test[TARGET], prediction, zero_division=0)),
        "recall": float(recall_score(test[TARGET], prediction, zero_division=0)),
        "f1": float(f1_score(test[TARGET], prediction, zero_division=0)),
        "confusion_matrix": matrix.tolist(), "test_size": int(len(test)),
    }
    scored = test[["player_id", TARGET]].assign(churn_probability=probability, predicted_churn=prediction)
    return model, metrics, scored
