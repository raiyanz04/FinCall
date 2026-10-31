# Copy-in instructions for the FinCall quantitative extension

Two archives are provided:

- `FinCall-quant-enhanced.zip`: full project with the existing CALL-E app plus the quantitative extension.
- `FinCall-quant-addon.zip`: only the files to add or replace in your existing repository.

## Use the full project archive

1. Back up your existing repository.
2. Extract `FinCall-quant-enhanced.zip`.
3. Copy the extracted `FinCall-main` contents into your repository, preserving the folder structure.
4. Keep your own `.env` file and actual customer records outside version control. The archive contains only a placeholder `.env.example`.
5. Install the requirements and run the app as described in `README.md`.

## Use the addon archive instead

Extract `FinCall-quant-addon.zip` into your existing repository root and allow these files to overwrite their existing versions:

- `app.py`
- `README.md`
- `requirements.txt`
- `.gitignore`

Add all other files/directories at their matching relative paths. Do not overwrite your `.env` file. Existing `fincall.py`, invoice data, and CALL-E integration are retained.

## First run

```bash
python -m venv .venv
# activate the environment
pip install -r requirements.txt
python scripts/run_quant_demo.py
streamlit run app.py
```

The quantitative demonstration runs without a CALL-E key. A valid `CALLE_API_KEY` is needed only when you manually initiate a live phone call.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

The original repository's `test_call.py` was replaced by `manual_live_call.py`. It can place a real call, so it is deliberately not part of pytest and requires a phone number in `.env` plus an explicit typed confirmation.

## Data caution

The demo payment observations are synthetic. Do not present their outputs as real predictive performance. The real payment-observation CSV and call-event JSON are git-ignored to reduce the chance of accidentally publishing customer records. Add only authorized, anonymized or appropriately protected data.
