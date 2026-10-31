from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.cashflow import simulate_portfolio_collections
from quant.data import load_observations, save_observation
from quant.markov import estimate_transition_matrix
from quant.survival import WeibullAFTModel, concordance_index, kaplan_meier
from scripts.generate_demo_data import generate_demo_data


def fitted_model():
    frame = generate_demo_data(n=500, seed=123)
    model = WeibullAFTModel().fit(frame, frame.duration_days, frame.event_observed)
    return frame, model


def test_weibull_aft_fit_and_predictions():
    frame, model = fitted_model()
    assert model.shape_ > 0
    median = model.predict_median_days(frame.iloc[:8])
    assert median.shape == (8,)
    assert np.isfinite(median).all()
    assert (median > 0).all()
    survival = model.predict_survival(frame.iloc[:2], [7, 14, 30])
    assert survival.shape == (2, 3)
    assert np.all(np.diff(survival, axis=1) <= 1e-12)


def test_kaplan_meier_non_increasing():
    curve = kaplan_meier([2, 3, 4, 5], [1, 0, 1, 1])
    assert curve.iloc[0].survival_probability == 1.0
    assert np.all(np.diff(curve.survival_probability) <= 1e-12)


def test_concordance_index_bounds():
    score = concordance_index([1, 2, 3, 4], [1, 1, 1, 1], [1, 2, 3, 4])
    assert score == pytest.approx(1.0)


def test_monte_carlo_outputs_monotone_expected_collections():
    _, model = fitted_model()
    portfolio = pd.DataFrame({
        "amount": [10000.0, 20000.0, 5000.0],
        "prior_delay_days": [5.0, 20.0, 10.0],
        "follow_up_count": [1.0, 2.0, 0.0],
        "promised_flag": [1, 0, 0],
        "age_days": [10.0, 25.0, 4.0],
    })
    summary, samples = simulate_portfolio_collections(
        model, portfolio, n_simulations=1000, minimum_collection_target=15000, random_seed=7
    )
    assert list(summary.horizon_days) == [7, 14, 30]
    assert samples.shape == (1000, 3)
    assert summary.expected_collections.is_monotonic_increasing
    assert summary.shortfall_expected_shortfall.ge(summary.shortfall_var).all()


def test_markov_transition_estimation():
    events = [
        {"invoice_id": "A", "payment_status": "delayed", "called_at": "2026-01-01"},
        {"invoice_id": "A", "payment_status": "payment_promised", "called_at": "2026-01-02"},
        {"invoice_id": "A", "payment_status": "paid", "called_at": "2026-01-03"},
    ]
    matrix, count = estimate_transition_matrix(events, smoothing=0.0)
    assert count == 2
    assert matrix.loc["delayed", "payment_promised"] == pytest.approx(1.0)
    assert matrix.loc["payment_promised", "paid"] == pytest.approx(1.0)


def test_save_observation_round_trip(tmp_path: Path):
    path = tmp_path / "observations.csv"
    row = {
        "invoice_id": "INV-1", "customer_id": "C-1", "amount": 1000,
        "prior_delay_days": 2, "follow_up_count": 1, "promised_flag": 1,
        "duration_days": 9, "event_observed": 1,
        "observation_date": "2026-10-01", "data_source": "test",
    }
    save_observation(path, row)
    saved = load_observations(path)
    assert len(saved) == 1
    assert saved.iloc[0].invoice_id == "INV-1"


def test_fit_rejects_too_few_observations():
    frame = generate_demo_data(n=100, seed=4).head(15)
    with pytest.raises(ValueError):
        WeibullAFTModel().fit(frame, frame.duration_days, frame.event_observed)


def test_expected_shortfall_uses_upper_tail_when_var_is_zero():
    _, model = fitted_model()
    portfolio = pd.DataFrame({
        "amount": [1.0],
        "prior_delay_days": [0.0],
        "follow_up_count": [0.0],
        "promised_flag": [0],
        "age_days": [0.0],
    })
    summary, _ = simulate_portfolio_collections(
        model,
        portfolio,
        horizons=(30,),
        n_simulations=1000,
        minimum_collection_target=0.0,
        random_seed=9,
    )
    assert summary.iloc[0].shortfall_var == 0.0
    assert summary.iloc[0].shortfall_expected_shortfall == 0.0


def test_conditional_payment_probability_and_median():
    frame, model = fitted_model()
    sample = frame.iloc[:6].copy()
    ages = np.array([0, 5, 10, 20, 30, 40], dtype=float)
    median_remaining = model.predict_conditional_median_remaining(sample, ages)
    assert median_remaining.shape == (6,)
    assert (median_remaining > 0).all()
    probability = model.conditional_payment_probability(sample, ages, 7)
    assert probability.shape == (6,)
    assert ((probability >= 0) & (probability <= 1)).all()
    assert np.all(probability <= model.conditional_payment_probability(sample, ages, 30))
