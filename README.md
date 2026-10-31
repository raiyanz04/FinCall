# FinCall: AI Financial Operations + Quantitative Receivables Risk

FinCall is an AI-powered financial operations phone agent built with CALL-E. It prioritizes overdue invoices, conducts user-initiated payment follow-up calls, captures structured outcomes, and provides an experimental quantitative layer for payment-timing and cash-flow-risk analysis.

## Existing application

- Overdue invoice prioritization
- AI-powered outbound phone calls through CALL-E
- Structured payment-status and expected-date extraction
- Call evidence and recommended next actions
- Streamlit dashboard and persistent latest-call history

## Quantitative extension

The additional analytics module includes:

1. **Kaplan-Meier survival analysis** as a non-parametric payment-time baseline.
2. **Weibull accelerated-failure-time (AFT) regression** with right-censored observations and invoice-level covariates.
3. **Conditional payment-time simulation** for invoices that remain unpaid as of the current date.
4. **Monte Carlo portfolio cash-flow projections** over 7, 14, and 30-day horizons.
5. **Cash-shortfall VaR and Expected Shortfall** relative to a user-defined collection target.
6. **Call-outcome Markov transition estimates** from append-only CALL-E event history.
7. A form to record paid and still-unpaid observations so the model can transition from demonstration data to real observed data.

## Important data distinction

`data/demo_payment_observations.csv` is **synthetic demonstration data** generated from a known Weibull data-generating process. It is not real customer/payment history and must not be used to claim production accuracy.

`data/payment_observations.csv` is the template for genuine, authorized payment observations. One row represents one invoice episode at its final payment event or censoring date. A row with `event_observed=1` means payment occurred; `event_observed=0` means the invoice was still unpaid at the observation date (right-censored).

The application uses real observations only after there are at least 40 valid rows, including at least 8 paid and 8 censored observations. Otherwise it explicitly falls back to the synthetic demonstration dataset. These thresholds are safeguards for a teaching prototype, not statistical guarantees of model adequacy.

## Model assumptions and limitations

- The Weibull AFT model assumes a Weibull distribution for payment duration conditional on the included covariates.
- Current baseline Monte Carlo simulation assumes conditional independence between invoice payment times. It does **not** model common customer, sector, or macroeconomic shocks.
- The model estimates time from invoice due date. Existing invoices are simulated conditionally on remaining unpaid until today.
- `prior_delay_days`, `follow_up_count`, and `promised_flag` are simplified covariates. Better customer-level history and event timestamps are needed for a serious study.
- Small or synthetic samples cannot establish predictive validity. Evaluate on later time periods, check calibration, and compare against simple baselines before drawing conclusions.
- The Markov matrix is descriptive and uses Laplace smoothing. It should not be used to automate collection decisions with very small event histories.
- Do not use the model to make adverse decisions about customers. Keep human review for disputed invoices and sensitive interactions.

## Repository structure

```text
FinCall-main/
├── app.py
├── fincall.py
├── quant/
│   ├── cashflow.py
│   ├── data.py
│   ├── evaluation.py
│   ├── markov.py
│   └── survival.py
├── data/
│   ├── invoices.json
│   ├── call_history.json
│   ├── call_events.json
│   ├── payment_observations.csv
│   └── demo_payment_observations.csv
├── scripts/
│   └── generate_demo_data.py
└── tests/
    └── test_quantitative.py
```

## Setup

Python 3.11 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repository root:

```env
CALLE_API_KEY=your_calle_api_key
```

Run the app:

```bash
streamlit run app.py
```

The quantitative analytics panel does not require a CALL-E API key. A key is required only when you initiate a live phone call.

## Regenerate synthetic demonstration observations

```bash
python scripts/generate_demo_data.py --rows 1200 --seed 2026
```

The seed makes the demonstration dataset reproducible.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The tests cover model fitting, survival-curve monotonicity, concordance scoring, Monte Carlo outputs, Markov transitions, and observation persistence. The holdout evaluation prefers a chronological split by observation date when those dates are available.

Run a standalone quantitative demonstration without launching Streamlit or placing a CALL-E call:

```bash
python scripts/run_quant_demo.py
```

## Live-call safety

- Never commit `.env` or API credentials.
- Use only phone numbers for which you have explicit authorization.
- Live calls are initiated manually by the user.
- The `manual_live_call.py` script initiates a real outbound test call to the configured number. Do not run it as part of the test suite.
- Do not place real personal data in synthetic demo files or commit customer data to a public repository.
