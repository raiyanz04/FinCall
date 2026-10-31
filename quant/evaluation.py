"""Simple reproducible holdout evaluation for the Weibull AFT model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .survival import WeibullAFTModel, concordance_index


def evaluate_weibull_aft(frame: pd.DataFrame, test_fraction: float = 0.25, seed: int = 42) -> dict:
    if not 0.1 <= test_fraction <= 0.4:
        raise ValueError("test_fraction must be between 0.1 and 0.4")
    if len(frame) < 40:
        raise ValueError("At least 40 observations are required for holdout evaluation")
    # Prefer a chronological split to reduce future-to-past information leakage.
    observed_dates = pd.to_datetime(frame["observation_date"], errors="coerce") if "observation_date" in frame.columns else pd.Series(dtype="datetime64[ns]")
    if len(observed_dates) == len(frame) and observed_dates.notna().mean() >= 0.9:
        ordered = frame.assign(_split_date=observed_dates).sort_values("_split_date").drop(columns="_split_date")
        split = int(round(len(ordered) * (1.0 - test_fraction)))
        train = ordered.iloc[:split].reset_index(drop=True)
        test = ordered.iloc[split:].reset_index(drop=True)
    else:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(frame))
        split = int(round(len(frame) * (1.0 - test_fraction)))
        train = frame.iloc[indices[:split]].reset_index(drop=True)
        test = frame.iloc[indices[split:]].reset_index(drop=True)

    model = WeibullAFTModel().fit(
        train,
        train["duration_days"],
        train["event_observed"],
    )
    predicted_median = model.predict_median_days(test)
    c_index = concordance_index(test["duration_days"], test["event_observed"], predicted_median)
    return {
        "model": model,
        "train_rows": len(train),
        "test_rows": len(test),
        "test_paid": int(test["event_observed"].sum()),
        "test_censored": int((test["event_observed"] == 0).sum()),
        "c_index": c_index,
        "converged": model.converged_,
        "optimizer_message": model.message_,
    }
