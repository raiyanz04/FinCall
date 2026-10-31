"""Generate deterministic synthetic right-censored invoice payment observations.

These records are explicitly simulated and must never be presented as real
customer/payment history or as evidence of production model performance.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "demo_payment_observations.csv"


def generate_demo_data(n: int = 1200, seed: int = 2026) -> pd.DataFrame:
    if n < 100:
        raise ValueError("Generate at least 100 demo observations")
    rng = np.random.default_rng(seed)
    amount = np.clip(rng.lognormal(mean=10.7, sigma=0.85, size=n), 2_000, 2_000_000)
    prior_delay = np.clip(rng.gamma(shape=2.0, scale=8.0, size=n), 0, 120)
    followups = np.clip(rng.poisson(lam=1.4, size=n), 0, 8)

    # Synthetic call promise is correlated with customer payment behaviour.
    promise_probability = 1.0 / (1.0 + np.exp(-(-0.9 + 0.018 * prior_delay - 0.22 * followups)))
    promised = rng.binomial(1, promise_probability, size=n)

    # Known synthetic data-generating process: payment durations follow a
    # covariate-dependent Weibull AFT distribution.
    log_scale = (
        np.log(24.0)
        + 0.13 * (np.log1p(amount) - np.log1p(amount).mean()) / np.log1p(amount).std()
        + 0.27 * (prior_delay - prior_delay.mean()) / prior_delay.std()
        + 0.12 * (followups - followups.mean()) / (followups.std() or 1.0)
        - 0.32 * promised
    )
    shape = 1.35
    scale = np.exp(log_scale)
    payment_time = scale * np.power(-np.log(np.clip(rng.random(n), 1e-12, 1.0)), 1.0 / shape)

    # Administrative follow-up ends between 15 and 100 days after due date.
    censor_time = rng.integers(15, 101, size=n).astype(float)
    observed = (payment_time <= censor_time).astype(int)
    duration = np.maximum(1.0, np.minimum(payment_time, censor_time))

    # Keep all synthetic observations historical relative to the 2026 demo context.
    due_dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 500, size=n), unit="D")
    observation_dates = due_dates + pd.to_timedelta(np.ceil(duration).astype(int), unit="D")
    return pd.DataFrame({
        "invoice_id": [f"DEMO-{i:05d}" for i in range(1, n + 1)],
        "customer_id": [f"DEMO-CUST-{rng.integers(1, 301):04d}" for _ in range(n)],
        "amount": np.round(amount, 2),
        "prior_delay_days": np.round(prior_delay, 2),
        "follow_up_count": followups,
        "promised_flag": promised,
        "duration_days": np.round(duration, 3),
        "event_observed": observed,
        "observation_date": observation_dates.strftime("%Y-%m-%d"),
        "data_source": "synthetic_demo",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = generate_demo_data(n=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Wrote {len(data):,} synthetic observations to {args.output}")
    print(f"Paid events: {int(data.event_observed.sum()):,}; censored: {int((data.event_observed == 0).sum()):,}")


if __name__ == "__main__":
    main()
