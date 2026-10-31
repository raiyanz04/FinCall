# FinCall quantitative model card

## Intended use

An educational prototype for studying invoice payment timing, survival analysis, cash-flow simulation, and shortfall-risk summaries. It is not a credit-scoring, collections prioritisation, lending, or automated adverse-decision system.

## Models

- Kaplan-Meier survival estimator as a non-parametric baseline.
- Weibull accelerated-failure-time regression fitted by maximum likelihood with right-censored observations.
- Conditional Monte Carlo simulation for future portfolio collections.
- Smoothed descriptive Markov transition matrix for repeated CALL-E outcome states.

## Inputs

Invoice amount, prior average payment delay, number of follow-up calls, payment-promise indicator, time from due date to payment or censoring, and event indicator.

## Data provenance

The committed sample `data/demo_payment_observations.csv` is synthetic and generated with a fixed random seed and known Weibull process. Real observed data must be entered in `data/payment_observations.csv` and should be authorized, privacy-reviewed, and protected from accidental public publication.

## Evaluation

A chronological holdout is used when observation dates are available. Concordance measures ranking of payment times under censoring; it does not measure probability calibration. Research-grade work should additionally report calibration, Brier scores with appropriate censoring weights, confidence intervals, sensitivity analysis, time-based out-of-sample performance, and comparisons with simple baselines.

## Known limitations

- Weibull duration assumption may not fit multimodal, seasonal, or highly non-stationary payment processes.
- Conditional invoice independence ignores common shocks, customer groups, and sector-level dependence.
- The included covariates are intentionally simple and may be insufficient for real payment behaviour.
- Historical call outcomes may be sparse, selectively observed, or affected by intervention bias.
- Synthetic training results do not establish real-world predictive validity.
- Currency amounts must be consistently denominated; mixed currencies require conversion or explicit currency treatment.

## Human oversight and privacy

Do not use these outputs to automatically deny service, apply penalties, or make adverse decisions about customers. Disputed invoices and uncertain outcomes require human review. Do not commit credentials or personal customer information.
