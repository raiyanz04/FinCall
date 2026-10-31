"""Survival-analysis models for invoice payment timing.

The Weibull AFT model below supports right-censored observations. It is a
compact, transparent baseline intended for learning and experimentation,
not a production credit decision engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

FEATURE_COLUMNS = [
    "amount",
    "prior_delay_days",
    "follow_up_count",
    "promised_flag",
]


def prepare_features(frame: pd.DataFrame) -> np.ndarray:
    """Return the model's unscaled numeric feature matrix."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing survival-model feature columns: {missing}")

    x = frame[FEATURE_COLUMNS].copy()
    x["amount"] = np.log1p(pd.to_numeric(x["amount"], errors="coerce").clip(lower=0))
    for column in FEATURE_COLUMNS[1:]:
        x[column] = pd.to_numeric(x[column], errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x.to_numpy(dtype=float)


@dataclass
class WeibullAFTModel:
    """Weibull accelerated-failure-time regression with right censoring.

    ``scale(x) = exp(intercept + standardized_x @ coefficients)`` and
    ``S(t|x) = exp(-(t / scale(x)) ** shape)``.
    """

    l2_penalty: float = 1e-4
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    coefficients_: np.ndarray | None = None
    intercept_: float | None = None
    shape_: float | None = None
    converged_: bool = False
    message_: str = "Not fitted"

    def fit(
        self,
        features: pd.DataFrame,
        durations: Iterable[float],
        event_observed: Iterable[int],
    ) -> "WeibullAFTModel":
        raw_x = prepare_features(features)
        t = np.asarray(list(durations), dtype=float)
        event = np.asarray(list(event_observed), dtype=int)

        if raw_x.ndim != 2 or len(raw_x) != len(t) or len(t) != len(event):
            raise ValueError("Features, durations, and event indicators must have matching rows")
        if len(t) < 20:
            raise ValueError("At least 20 observations are required to fit the Weibull AFT model")
        if not np.isfinite(t).all() or np.any(t <= 0):
            raise ValueError("Durations must be finite and greater than zero")
        if not np.isin(event, [0, 1]).all():
            raise ValueError("event_observed must contain only 0 (censored) or 1 (paid)")
        if event.sum() < 5 or (1 - event).sum() < 5:
            raise ValueError("Need at least five paid and five right-censored observations")

        self.mean_ = raw_x.mean(axis=0)
        std = raw_x.std(axis=0)
        self.scale_ = np.where(std < 1e-8, 1.0, std)
        x = (raw_x - self.mean_) / self.scale_
        log_t = np.log(np.maximum(t, 1e-6))

        # Parameter vector: intercept, feature coefficients, log(shape).
        initial = np.zeros(x.shape[1] + 2, dtype=float)
        initial[0] = float(np.median(log_t))
        initial[-1] = np.log(1.2)

        def negative_log_likelihood(params: np.ndarray) -> float:
            intercept = params[0]
            beta = params[1:-1]
            shape = np.exp(np.clip(params[-1], -3.0, 3.0))
            log_lambda = np.clip(intercept + x @ beta, -8.0, 12.0)
            z = np.exp(np.clip(shape * (log_t - log_lambda), -50.0, 50.0))

            # log S(t) for censored observations; log f(t) for paid ones.
            log_likelihood = -z
            paid = event == 1
            log_likelihood[paid] += (
                np.log(shape) - log_t[paid]
                + shape * (log_t[paid] - log_lambda[paid])
            )
            penalty = self.l2_penalty * float(np.dot(beta, beta))
            value = -float(np.sum(log_likelihood)) + penalty
            return value if np.isfinite(value) else 1e100

        result = minimize(
            negative_log_likelihood,
            initial,
            method="L-BFGS-B",
            bounds=[(None, None)] * (len(initial) - 1) + [(-3.0, 3.0)],
            options={"maxiter": 2000, "ftol": 1e-10},
        )

        if not np.isfinite(result.fun):
            raise RuntimeError("Weibull AFT optimisation returned a non-finite objective")

        self.intercept_ = float(result.x[0])
        self.coefficients_ = np.asarray(result.x[1:-1], dtype=float)
        self.shape_ = float(np.exp(result.x[-1]))
        self.converged_ = bool(result.success)
        self.message_ = str(result.message)
        return self

    def _check_fitted(self) -> None:
        if any(value is None for value in (self.mean_, self.scale_, self.coefficients_, self.intercept_, self.shape_)):
            raise RuntimeError("Fit the WeibullAFTModel before requesting predictions")

    def predict_scale(self, features: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        raw_x = prepare_features(features)
        if raw_x.shape[1] != len(self.coefficients_):
            raise ValueError("Prediction feature count does not match fitted model")
        x = (raw_x - self.mean_) / self.scale_
        log_lambda = np.clip(self.intercept_ + x @ self.coefficients_, -8.0, 12.0)
        return np.exp(log_lambda)

    def predict_survival(self, features: pd.DataFrame, times: Iterable[float]) -> np.ndarray:
        scale = self.predict_scale(features)
        t = np.asarray(list(times), dtype=float)
        if np.any(t < 0):
            raise ValueError("Prediction times cannot be negative")
        return np.exp(-np.power(t[None, :] / scale[:, None], self.shape_))

    def predict_median_days(self, features: pd.DataFrame) -> np.ndarray:
        """Unconditional median duration measured from the invoice due date."""
        scale = self.predict_scale(features)
        return scale * np.power(np.log(2.0), 1.0 / self.shape_)

    def predict_conditional_median_remaining(
        self,
        features: pd.DataFrame,
        current_age_days: Iterable[float],
    ) -> np.ndarray:
        """Median additional days until payment, conditional on still being unpaid."""
        scale = self.predict_scale(features)
        age = np.asarray(list(current_age_days), dtype=float)
        if len(age) != len(scale) or np.any(age < 0):
            raise ValueError("current_age_days must be non-negative and match feature rows")
        absolute_time = scale * np.power(
            np.power(age / scale, self.shape_) + np.log(2.0),
            1.0 / self.shape_,
        )
        return np.maximum(absolute_time - age, 0.0)

    def conditional_payment_probability(
        self,
        features: pd.DataFrame,
        current_age_days: Iterable[float],
        horizon_days: float,
    ) -> np.ndarray:
        """Probability of payment within the next horizon, given survival to age."""
        if horizon_days < 0:
            raise ValueError("horizon_days cannot be negative")
        scale = self.predict_scale(features)
        age = np.asarray(list(current_age_days), dtype=float)
        if len(age) != len(scale) or np.any(age < 0):
            raise ValueError("current_age_days must be non-negative and match feature rows")
        cumulative_hazard_increment = (
            np.power((age + horizon_days) / scale, self.shape_)
            - np.power(age / scale, self.shape_)
        )
        return np.clip(-np.expm1(-cumulative_hazard_increment), 0.0, 1.0)

    def sample_remaining_days(
        self,
        features: pd.DataFrame,
        current_age_days: Iterable[float],
        n_simulations: int = 10_000,
        random_seed: int = 42,
    ) -> np.ndarray:
        """Sample conditional time-to-payment given each invoice is still open.

        Returns an array shaped (n_simulations, n_invoices).
        """
        scale = self.predict_scale(features)
        age = np.asarray(list(current_age_days), dtype=float)
        if len(age) != len(scale) or np.any(age < 0):
            raise ValueError("current_age_days must be non-negative and match feature rows")
        if n_simulations < 100:
            raise ValueError("Use at least 100 simulations")

        rng = np.random.default_rng(random_seed)
        u = np.clip(rng.random((n_simulations, len(scale))), 1e-12, 1.0 - 1e-12)
        scaled_age_power = np.power(age[None, :] / scale[None, :], self.shape_)
        next_scaled_power = scaled_age_power - np.log(u)
        absolute_payment_time = scale[None, :] * np.power(next_scaled_power, 1.0 / self.shape_)
        remaining = absolute_payment_time - age[None, :]
        return np.maximum(remaining, 0.0)


def kaplan_meier(durations: Iterable[float], event_observed: Iterable[int]) -> pd.DataFrame:
    """Compute a Kaplan-Meier survival curve, including right censoring."""
    t = np.asarray(list(durations), dtype=float)
    event = np.asarray(list(event_observed), dtype=int)
    if len(t) != len(event) or len(t) == 0:
        raise ValueError("Durations and event indicators must be non-empty and aligned")
    if np.any(t <= 0) or not np.isfinite(t).all() or not np.isin(event, [0, 1]).all():
        raise ValueError("Invalid survival observations")

    survival = 1.0
    rows = [{"day": 0.0, "survival_probability": 1.0, "at_risk": len(t), "events": 0}]
    for day in np.sort(np.unique(t[event == 1])):
        at_risk = int(np.sum(t >= day))
        events = int(np.sum((t == day) & (event == 1)))
        if at_risk:
            survival *= 1.0 - events / at_risk
        rows.append({"day": float(day), "survival_probability": survival, "at_risk": at_risk, "events": events})
    return pd.DataFrame(rows)


def concordance_index(durations: Iterable[float], event_observed: Iterable[int], risk_score: Iterable[float]) -> float:
    """Harrell-style concordance index for predicted time/risk ordering.

    A smaller predicted payment time means higher risk / earlier payment.
    """
    t = np.asarray(list(durations), dtype=float)
    event = np.asarray(list(event_observed), dtype=int)
    prediction = np.asarray(list(risk_score), dtype=float)
    if not (len(t) == len(event) == len(prediction)):
        raise ValueError("Concordance inputs must have equal lengths")
    concordant = 0.0
    comparable = 0
    for i in range(len(t)):
        if event[i] != 1:
            continue
        for j in range(len(t)):
            if t[i] < t[j]:
                comparable += 1
                if prediction[i] < prediction[j]:
                    concordant += 1
                elif prediction[i] == prediction[j]:
                    concordant += 0.5
    return float(concordant / comparable) if comparable else float("nan")
