# Quantitative methodology

## 1. Payment duration and censoring

For invoice episode i, define T_i as the number of days from the due date to the payment date. If payment has not occurred by the last observation date, the duration is right-censored. The event indicator δ_i is 1 for observed payment and 0 for a censored invoice.

## 2. Kaplan-Meier baseline

The Kaplan-Meier estimator is

S_hat(t) = product over event times u <= t of (1 - d_u / n_u),

where d_u is the number of payments at u and n_u is the number of invoices still at risk immediately before u. Censored observations leave the risk set after their censoring time but do not count as payment events.

## 3. Weibull AFT model

The conditional survival function is

S(t | x) = exp(-(t / λ(x))^ρ),

where ρ > 0 is the Weibull shape and log λ(x) = β_0 + x'β. The covariates are log(1 + invoice amount), prior average payment delay, follow-up count, and a payment-promise indicator. Parameters are fitted by maximum likelihood using both payment events and right-censored observations, with a small L2 penalty on feature coefficients.

For a currently unpaid invoice with age a, the probability of payment during the next h days is

P(a < T <= a+h | T > a, x) = 1 - S(a+h | x) / S(a | x).

The conditional median remaining time r_0.5 solves S(a+r_0.5 | x) / S(a | x) = 0.5, giving

r_0.5 = λ [ (a/λ)^ρ + log(2) ]^(1/ρ) - a.

## 4. Conditional Monte Carlo simulation

For each open invoice, sample U ~ Uniform(0,1) and invert the conditional Weibull survival distribution:

T_remaining = λ [ (a/λ)^ρ - log(U) ]^(1/ρ) - a.

For portfolio horizon h, simulated collections are

C_h = sum_i A_i 1(T_remaining,i <= h),

where A_i is the outstanding invoice amount. The current implementation assumes conditional independence across invoices.

## 5. Shortfall risk

For a minimum collection target K, define shortfall L_h = max(0, K - C_h). Shortfall VaR at confidence q is the q-quantile of L_h. Expected Shortfall is the average simulated shortfall at or beyond that VaR threshold.

## 6. Validation

The application reports a deterministic random holdout concordance index. This is a ranking metric, not a calibration measure and not a substitute for time-based validation. For research-quality evaluation, use chronological train/test splits, calibration plots, Brier scores or integrated Brier scores, confidence intervals, sensitivity analysis, and comparison with Kaplan-Meier or simple empirical baselines.

## 7. Synthetic data

The committed demo observations are simulated with a known Weibull data-generating process and random administrative censoring. They are included to make the application demonstrable before enough real invoice outcomes have accumulated. All dashboard outputs based on these observations are synthetic demonstrations only.
