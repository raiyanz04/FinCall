"""Data loading and validation helpers for quantitative FinCall analytics."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

OBSERVATION_COLUMNS = [
    "invoice_id",
    "customer_id",
    "amount",
    "prior_delay_days",
    "follow_up_count",
    "promised_flag",
    "duration_days",
    "event_observed",
    "observation_date",
    "data_source",
]


def load_json_list(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def load_observations(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)
    frame = pd.read_csv(file_path)
    missing = [column for column in OBSERVATION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Observation file is missing columns: {missing}")
    return validate_observations(frame)


def validate_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the observation table and reject invalid or ambiguous rows."""
    missing = [column for column in OBSERVATION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Observation data is missing required columns: {missing}")
    clean = frame[OBSERVATION_COLUMNS].copy()
    for column in ["amount", "prior_delay_days", "follow_up_count", "promised_flag", "duration_days", "event_observed"]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.dropna(subset=["amount", "prior_delay_days", "follow_up_count", "promised_flag", "duration_days", "event_observed"])
    clean = clean[(clean["amount"] >= 0) & (clean["prior_delay_days"] >= 0) & (clean["follow_up_count"] >= 0)]
    clean = clean[(clean["duration_days"] > 0) & clean["event_observed"].isin([0, 1]) & clean["promised_flag"].isin([0, 1])]
    clean["invoice_id"] = clean["invoice_id"].astype(str)
    clean["customer_id"] = clean["customer_id"].astype(str)
    clean["data_source"] = clean["data_source"].astype(str)
    return clean.reset_index(drop=True)


def select_training_data(
    real_path: str | Path,
    demo_path: str | Path,
    minimum_rows: int = 40,
    minimum_events_each_class: int = 8,
) -> tuple[pd.DataFrame, str, str]:
    """Prefer sufficiently rich real records; otherwise use labelled demo data."""
    real = load_observations(real_path)
    paid = int(real["event_observed"].sum()) if not real.empty else 0
    censored = int((real["event_observed"] == 0).sum()) if not real.empty else 0
    if len(real) >= minimum_rows and paid >= minimum_events_each_class and censored >= minimum_events_each_class:
        return real, "Observed payment history", "real"

    demo = load_observations(demo_path)
    if demo.empty:
        raise ValueError("No valid real or demonstration payment observations are available")
    reason = (
        f"Real history has {len(real)} usable rows ({paid} paid, {censored} censored); "
        f"at least {minimum_rows} rows and {minimum_events_each_class} of each outcome are required. "
        "The model is therefore using synthetic demonstration observations."
    )
    return demo, reason, "synthetic"


def save_observation(path: str | Path, row: dict[str, Any]) -> None:
    """Insert or replace one terminal/censored observation per invoice episode."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    current = load_observations(file_path)
    new_row = pd.DataFrame([row], columns=OBSERVATION_COLUMNS)
    combined = pd.concat([current, new_row], ignore_index=True)
    combined = validate_observations(combined)
    if not combined.empty:
        # One final observation per invoice episode. Re-entering an invoice updates it.
        combined = combined.drop_duplicates(subset=["invoice_id"], keep="last")
    combined.to_csv(file_path, index=False)


def build_open_invoice_features(
    invoices: list[dict[str, Any]],
    call_history: dict[str, dict[str, Any]],
    call_events: list[dict[str, Any]],
    today,
) -> pd.DataFrame:
    """Create model features for invoices that are not marked paid."""
    event_counts: dict[str, int] = {}
    for event in call_events:
        invoice_id = str(event.get("invoice_id", ""))
        if invoice_id:
            event_counts[invoice_id] = event_counts.get(invoice_id, 0) + 1

    rows = []
    for invoice in invoices:
        if str(invoice.get("status", "overdue")).lower() == "paid":
            continue
        due = pd.to_datetime(invoice["due_date"], errors="coerce")
        if pd.isna(due):
            continue
        age_days = max(0, int((pd.Timestamp(today) - due.normalize()).days))
        invoice_id = str(invoice["invoice_id"])
        latest = call_history.get(invoice_id, {})
        status = str(latest.get("payment_status", "unknown")).lower()
        rows.append({
            "invoice_id": invoice_id,
            "customer": invoice.get("customer", "Unknown"),
            "amount": float(invoice.get("amount", 0)),
            "prior_delay_days": float(invoice.get("prior_delay_days", 0)),
            "follow_up_count": float(invoice.get("follow_up_count", event_counts.get(invoice_id, 0))),
            "promised_flag": int(status == "payment_promised"),
            "age_days": float(age_days),
            "payment_status": status,
        })
    return pd.DataFrame(rows)
