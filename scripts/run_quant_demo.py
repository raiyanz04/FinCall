"""Run the quantitative FinCall demo without Streamlit or live phone calls."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.cashflow import simulate_portfolio_collections
from quant.data import build_open_invoice_features, load_json_list, load_observations
from quant.evaluation import evaluate_weibull_aft
from quant.survival import WeibullAFTModel


def main() -> None:
    demo = load_observations(ROOT / "data" / "demo_payment_observations.csv")
    evaluation = evaluate_weibull_aft(demo)
    model = WeibullAFTModel().fit(demo, demo["duration_days"], demo["event_observed"])
    print("FinCall quantitative demonstration")
    print("----------------------------------")
    print(f"Training observations: {len(demo):,}")
    print(f"Paid events: {int(demo.event_observed.sum()):,}")
    print(f"Right-censored: {int((demo.event_observed == 0).sum()):,}")
    print(f"Holdout concordance index: {evaluation['c_index']:.3f}")
    print(f"Weibull shape: {model.shape_:.3f}")
    print("NOTE: all model training data in this demo are synthetic.\n")

    invoices = json.loads((ROOT / "data" / "invoices.json").read_text(encoding="utf-8"))
    history_path = ROOT / "data" / "call_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else {}
    call_events = load_json_list(ROOT / "data" / "call_events.json")
    portfolio = build_open_invoice_features(invoices, history, call_events, today=date.today())
    if portfolio.empty:
        print("No open invoices found in data/invoices.json.")
        return

    summary, _ = simulate_portfolio_collections(
        model,
        portfolio,
        horizons=(7, 14, 30),
        n_simulations=10_000,
        minimum_collection_target=float(portfolio["amount"].sum() * 0.5),
        confidence_level=0.95,
        random_seed=42,
    )
    print(summary.to_string(index=False, formatters={
        "expected_collections": "₹{:,.0f}".format,
        "p05_collections": "₹{:,.0f}".format,
        "median_collections": "₹{:,.0f}".format,
        "p95_collections": "₹{:,.0f}".format,
        "probability_below_target": "{:.1%}".format,
        "shortfall_var": "₹{:,.0f}".format,
        "shortfall_expected_shortfall": "₹{:,.0f}".format,
    }))


if __name__ == "__main__":
    main()
