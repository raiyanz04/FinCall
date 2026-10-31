"""Descriptive Markov transition estimates from append-only call outcomes."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

STATES = ["unknown", "payment_promised", "delayed", "disputed", "paid"]


def estimate_transition_matrix(events: Iterable[dict], smoothing: float = 1.0) -> tuple[pd.DataFrame, int]:
    """Estimate smoothed within-invoice call-outcome transitions.

    Rows are the previous outcome and columns the next outcome. The function
    returns the matrix and the number of observed transitions. With little
    history this is descriptive only and should not drive customer treatment.
    """
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
    grouped: dict[str, list[tuple[str, str]]] = {}
    for item in events:
        invoice_id = str(item.get("invoice_id", "")).strip()
        status = str(item.get("payment_status", "unknown")).strip().lower()
        if invoice_id and status in STATES:
            grouped.setdefault(invoice_id, []).append((str(item.get("called_at", "")), status))

    counts = np.zeros((len(STATES), len(STATES)), dtype=float)
    state_index = {state: idx for idx, state in enumerate(STATES)}
    transitions = 0
    for history in grouped.values():
        history.sort(key=lambda pair: pair[0])
        statuses = [status for _, status in history]
        for old, new in zip(statuses, statuses[1:]):
            counts[state_index[old], state_index[new]] += 1
            transitions += 1

    if smoothing:
        counts += smoothing
    row_totals = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)
    return pd.DataFrame(probabilities, index=STATES, columns=STATES), transitions
