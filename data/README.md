# Data files

- `invoices.json`: current invoice records used by the original FinCall dashboard.
- `call_history.json`: latest CALL-E outcome per invoice, retained for dashboard compatibility.
- `call_events.json`: append-only CALL-E outcome history, used for descriptive transition analysis.
- `payment_observations.csv`: template for real, authorized invoice payment episodes.
- `demo_payment_observations.csv`: simulated demonstration data only; do not present it as real history.

For `payment_observations.csv`, each row should represent one invoice episode:

- `duration_days`: days from due date to payment or last observation date.
- `event_observed`: `1` if payment occurred; `0` if unpaid at last observation (right-censored).
- `amount`: invoice amount in the same currency units across the dataset.
- `prior_delay_days`: historical customer payment-delay feature known at the time of prediction.
- `follow_up_count`: calls made up to the prediction/observation point.
- `promised_flag`: `1` if a payment promise was recorded, otherwise `0`.
- `data_source`: provenance label such as `observed_user_entry` or `imported_accounting_system`.

Do not mix currencies in one model dataset without normalising amounts or adding currency-specific modelling. Do not upload personal or confidential customer records to a public repository.
