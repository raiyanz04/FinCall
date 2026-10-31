"""Monte Carlo portfolio collections and cash-shortfall risk metrics."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .survival import WeibullAFTModel


def simulate_portfolio_collections(
    model: WeibullAFTModel,
    invoices: pd.DataFrame,
    horizons: Iterable[int] = (7, 14, 30),
    n_simulations: int = 10_000,
    minimum_collection_target: float = 0.0,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Simulate invoice collection amounts over future horizons.

    Assumption: invoice payment times are conditionally independent given
    their model features. This baseline does not model customer-level or
    macroeconomic dependence.
    """
    required = {"amount", "prior_delay_days", "follow_up_count", "promised_flag", "age_days"}
    missing = sorted(required.difference(invoices.columns))
    if missing:
        raise ValueError(f"Missing portfolio columns: {missing}")
    if invoices.empty:
        raise ValueError("At least one open invoice is required")
    if minimum_collection_target < 0:
        raise ValueError("minimum_collection_target cannot be negative")
    if not 0.5 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0.5 and 1")

    amounts = pd.to_numeric(invoices["amount"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or np.any(amounts < 0):
        raise ValueError("Invoice amounts must be finite and non-negative")

    remaining_days = model.sample_remaining_days(
        invoices,
        invoices["age_days"].astype(float),
        n_simulations=n_simulations,
        random_seed=random_seed,
    )
    results = []
    for horizon in sorted(set(int(value) for value in horizons)):
        if horizon <= 0:
            raise ValueError("All horizons must be positive integers")
        collected = (remaining_days <= horizon) @ amounts
        shortfall = np.maximum(minimum_collection_target - collected, 0.0)
        var = float(np.quantile(shortfall, confidence_level))
        # Average the worst (1 - confidence_level) fraction of outcomes.
        # This remains meaningful when VaR is exactly zero.
        tail_count = max(1, int(np.ceil((1.0 - confidence_level) * len(shortfall))))
        tail = np.partition(shortfall, len(shortfall) - tail_count)[-tail_count:]
        expected_shortfall = float(tail.mean())
        results.append({
            "horizon_days": horizon,
            "expected_collections": float(collected.mean()),
            "p05_collections": float(np.quantile(collected, 0.05)),
            "median_collections": float(np.quantile(collected, 0.50)),
            "p95_collections": float(np.quantile(collected, 0.95)),
            "probability_below_target": float(np.mean(collected < minimum_collection_target)),
            "shortfall_var": var,
            "shortfall_expected_shortfall": expected_shortfall,
        })
    return pd.DataFrame(results), remaining_days
