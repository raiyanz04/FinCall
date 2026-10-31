import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from fincall import FinCall
from quant.cashflow import simulate_portfolio_collections
from quant.data import (
    build_open_invoice_features,
    load_json_list,
    save_observation,
    select_training_data,
)
from quant.evaluation import evaluate_weibull_aft
from quant.markov import estimate_transition_matrix
from quant.survival import WeibullAFTModel, kaplan_meier


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FinCall",
    page_icon="📞",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>
.fincall-title {
    font-size: 44px;
    font-weight: 700;
    margin-bottom: 0;
}

.fincall-subtitle {
    color: #9ca3af;
    font-size: 17px;
    margin-top: 4px;
}

.section-label {
    font-size: 13px;
    font-weight: 600;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}

.priority-high {
    color: #ff6b6b;
    font-weight: 700;
}

.priority-medium {
    color: #f59e0b;
    font-weight: 700;
}

.priority-low {
    color: #22c55e;
    font-weight: 700;
}

.invoice-card {
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #30363d;
    background: #161b22;
}

.history-card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #30363d;
    background: #161b22;
    margin-top: 10px;
}

.next-action {
    padding: 18px;
    border-left: 4px solid #3b82f6;
    background: #111827;
    border-radius: 8px;
    margin-top: 18px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path("data/invoices.json")
HISTORY_PATH = Path("data/call_history.json")
CALL_EVENTS_PATH = Path("data/call_events.json")
OBSERVATIONS_PATH = Path("data/payment_observations.csv")
DEMO_OBSERVATIONS_PATH = Path("data/demo_payment_observations.csv")


# ============================================================
# DATA FUNCTIONS
# ============================================================

def load_invoices():
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_call_history():
    if not HISTORY_PATH.exists():
        return {}

    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def save_call_history(history):
    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def calculate_priority(invoice):
    """
    Calculate collection priority using:
    - Outstanding amount
    - Number of days overdue
    """

    due_date = datetime.strptime(
        invoice["due_date"],
        "%Y-%m-%d"
    ).date()

    today = date.today()

    days_overdue = max(
        0,
        (today - due_date).days
    )

    amount = float(invoice["amount"])

    # Normalize amount roughly to a 0-100 range.
    amount_score = min(
        100,
        (amount / 150000) * 100
    )

    # More overdue days = higher priority.
    overdue_score = min(
        100,
        days_overdue * 15
    )

    # Amount is weighted slightly more heavily.
    score = (
        amount_score * 0.6
        + overdue_score * 0.4
    )

    if score >= 65:
        priority = "HIGH"
    elif score >= 35:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    return score, priority, days_overdue


# ============================================================
# LOAD DATA
# ============================================================

try:
    invoices = load_invoices()
    call_history = load_call_history()

except Exception as error:
    st.error(f"Could not load application data: {error}")
    st.stop()


# ============================================================
# RANK INVOICES
# ============================================================

ranked_invoices = []

for invoice in invoices:

    score, priority, days_overdue = calculate_priority(
        invoice
    )

    invoice_copy = invoice.copy()

    invoice_copy["priority_score"] = score
    invoice_copy["priority"] = priority
    invoice_copy["days_overdue"] = days_overdue

    ranked_invoices.append(invoice_copy)


ranked_invoices.sort(
    key=lambda x: x["priority_score"],
    reverse=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="fincall-title">📞 FinCall</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="fincall-subtitle">'
    'AI-powered financial operations phone agent'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# TOP METRICS
# ============================================================

total_overdue = sum(
    invoice["amount"]
    for invoice in ranked_invoices
)

high_priority_count = sum(
    1
    for invoice in ranked_invoices
    if invoice["priority"] == "HIGH"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Overdue Invoices",
        len(ranked_invoices)
    )

with col2:
    st.metric(
        "Outstanding Amount",
        f"₹{total_overdue:,.0f}"
    )

with col3:
    st.metric(
        "High Priority",
        high_priority_count
    )


st.divider()


# ============================================================
# COLLECTION PRIORITY
# ============================================================

st.markdown(
    '<div class="section-label">'
    'COLLECTION PRIORITY'
    '</div>',
    unsafe_allow_html=True,
)

st.subheader("Who should be called first?")


for index, invoice in enumerate(ranked_invoices):

    priority = invoice["priority"]

    if priority == "HIGH":
        icon = "🔴"
        priority_class = "priority-high"

    elif priority == "MEDIUM":
        icon = "🟠"
        priority_class = "priority-medium"

    else:
        icon = "🟢"
        priority_class = "priority-low"

    # Show latest call outcome if available.
    history = call_history.get(
        invoice["invoice_id"],
        {}
    )

    last_status = history.get(
        "payment_status",
        "Not contacted"
    )

    if last_status != "Not contacted":
        last_status = last_status.replace(
            "_",
            " "
        ).title()

    st.markdown(
        f"""
**{icon} {invoice["customer"]}**  
Invoice `{invoice["invoice_id"]}` ·
₹{invoice["amount"]:,.0f} ·
{invoice["days_overdue"]} day(s) overdue ·
<span class="{priority_class}">{priority} PRIORITY</span>  

Last outcome: **{last_status}**
""",
        unsafe_allow_html=True,
    )

    if index < len(ranked_invoices) - 1:
        st.divider()


st.divider()


# ============================================================
# SELECT CUSTOMER
# ============================================================

st.markdown(
    '<div class="section-label">'
    'AI PAYMENT FOLLOW-UP'
    '</div>',
    unsafe_allow_html=True,
)

invoice_options = {
    f'{invoice["invoice_id"]} · {invoice["customer"]} · '
    f'₹{invoice["amount"]:,.0f}': invoice["invoice_id"]
    for invoice in ranked_invoices
}


selected_label = st.selectbox(
    "Select an invoice to contact",
    options=list(invoice_options.keys()),
)

selected_invoice_id = invoice_options[selected_label]

selected_invoice = next(
    invoice
    for invoice in ranked_invoices
    if invoice["invoice_id"] == selected_invoice_id
)


# ============================================================
# SELECTED INVOICE
# ============================================================

st.markdown(
    f"""
<div class="invoice-card">
    <div class="section-label">SELECTED INVOICE</div>
    <h3>{selected_invoice["customer"]}</h3>
    <p>
        Invoice {selected_invoice["invoice_id"]} ·
        Due {selected_invoice["due_date"]}
    </p>
    <h2>₹{selected_invoice["amount"]:,.0f}</h2>
    <p>
        {selected_invoice["days_overdue"]} day(s) overdue ·
        <b>{selected_invoice["priority"]} PRIORITY</b>
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PREVIOUS CALL HISTORY
# ============================================================

previous_call = call_history.get(
    selected_invoice_id
)

if previous_call:

    st.write("")

    st.markdown(
        '<div class="section-label">'
        'PREVIOUS CALL'
        '</div>',
        unsafe_allow_html=True,
    )

    status = previous_call.get(
        "payment_status",
        "unknown"
    )

    status_display = status.replace(
        "_",
        " "
    ).title()

    expected_payment = previous_call.get(
        "expected_payment_date",
        "Unknown"
    )

    follow_up = previous_call.get(
        "follow_up_required",
        False
    )

    call_time = previous_call.get(
        "called_at",
        "Unknown"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Payment Status",
            status_display
        )

    with col2:
        st.metric(
            "Expected Payment",
            expected_payment
        )

    with col3:
        st.metric(
            "Follow-up",
            "Required"
            if follow_up
            else "Not Required"
        )

    st.caption(
        f"Last contacted: {call_time}"
    )


st.write("")


# ============================================================
# CALL CUSTOMER
# ============================================================

if st.button(
    "📞  Start AI Payment Follow-up",
    type="primary",
    use_container_width=True,
):

    fincall = FinCall()

    with st.spinner(
        "CALL-E is calling the customer..."
    ):

        try:

            result = fincall.call_customer(
                selected_invoice_id
            )

            st.success(
                "Call completed successfully"
            )

            structured = result.get(
                "structured_result",
                {}
            )

            payment_status = structured.get(
                "payment_status",
                "unknown"
            )

            expected_date = structured.get(
                "expected_payment_date",
                "Unknown"
            )

            follow_up = structured.get(
                "follow_up_required",
                False
            )

            evidence = result.get(
                "evidence",
                []
            )

            # ====================================================
            # SAVE CALL HISTORY
            # ====================================================

            call_history[selected_invoice_id] = {
                "payment_status": payment_status,
                "expected_payment_date": expected_date,
                "follow_up_required": follow_up,
                "evidence": evidence,
                "called_at": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            }

            save_call_history(call_history)

            # Keep an append-only event log for longitudinal analysis.
            call_events = load_json_list(CALL_EVENTS_PATH)
            call_events.append({
                "invoice_id": selected_invoice_id,
                "payment_status": payment_status,
                "expected_payment_date": expected_date,
                "follow_up_required": bool(follow_up),
                "called_at": datetime.now().isoformat(timespec="seconds"),
            })
            CALL_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with CALL_EVENTS_PATH.open("w", encoding="utf-8") as event_file:
                json.dump(call_events, event_file, indent=2)

            # ====================================================
            # CALL OUTCOME
            # ====================================================

            st.divider()

            st.markdown(
                '<div class="section-label">'
                'CALL OUTCOME'
                '</div>',
                unsafe_allow_html=True,
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Payment Status",
                    payment_status.replace(
                        "_",
                        " "
                    ).title(),
                )

            with col2:

                st.metric(
                    "Expected Payment",
                    expected_date,
                )

            with col3:

                st.metric(
                    "Follow-up",
                    "Required"
                    if follow_up
                    else "Not Required",
                )

            # ====================================================
            # AI EVIDENCE
            # ====================================================

            st.markdown(
                '<div class="section-label" '
                'style="margin-top:25px;">'
                'AI CALL EVIDENCE'
                '</div>',
                unsafe_allow_html=True,
            )

            if evidence:

                for item in evidence:

                    st.write(
                        "✓",
                        item
                    )

            else:

                st.write(
                    "No evidence returned."
                )

            # ====================================================
            # RECOMMENDED ACTION
            # ====================================================

            if payment_status == "payment_promised":

                next_action = (
                    "<b>Recommended next action</b>"
                    "<br><br>"
                    "Customer committed to payment. "
                    "Follow up after the promised payment "
                    "date if payment has not been confirmed."
                )

            elif payment_status == "delayed":

                next_action = (
                    "<b>Recommended next action</b>"
                    "<br><br>"
                    "Payment is delayed. "
                    "Schedule another follow-up and review "
                    "the customer's stated reason."
                )

            elif payment_status == "disputed":

                next_action = (
                    "<b>Recommended next action</b>"
                    "<br><br>"
                    "Customer disputed the invoice. "
                    "Route the case to the finance team "
                    "for manual review."
                )

            elif payment_status == "paid":

                next_action = (
                    "<b>Recommended next action</b>"
                    "<br><br>"
                    "Payment has been confirmed. "
                    "No further collection call is required."
                )

            else:

                next_action = (
                    "<b>Recommended next action</b>"
                    "<br><br>"
                    "Call outcome is inconclusive. "
                    "Schedule another attempt or manual review."
                )

            st.markdown(
                f"""
<div class="next-action">
{next_action}
</div>
""",
                unsafe_allow_html=True,
            )

            # ====================================================
            # REFRESH DATA
            # ====================================================

            st.info(
                "Call outcome saved to FinCall history."
            )

        except Exception as error:

            st.error(
                f"Call failed: {error}"
            )


# ============================================================
# QUANTITATIVE RECEIVABLES ANALYTICS
# ============================================================

st.divider()
st.markdown(
    '<div class="section-label">QUANTITATIVE RESEARCH EXTENSION</div>',
    unsafe_allow_html=True,
)
st.header("Receivables Risk & Stochastic Cash-Flow Modelling")
st.caption(
    "Experimental analytical layer: survival analysis estimates payment timing; "
    "Monte Carlo simulation translates those estimates into portfolio cash-flow risk."
)

try:
    training_data, training_note, training_source = select_training_data(
        OBSERVATIONS_PATH,
        DEMO_OBSERVATIONS_PATH,
    )
    if training_source == "synthetic":
        st.warning(
            "DEMONSTRATION MODE: the current model is trained on synthetic observations, "
            "not actual customer payment history. Record sufficient genuine outcomes in "
            "the form below before treating forecasts as data-driven business estimates.\n\n"
            + training_note
        )
    else:
        st.success(f"Training source: {training_note}")

    paid_count = int(training_data["event_observed"].sum())
    censored_count = int((training_data["event_observed"] == 0).sum())
    try:
        evaluation = evaluate_weibull_aft(training_data)
        c_index = evaluation["c_index"]
        evaluation_label = (
            f"Holdout concordance index: {c_index:.3f}" if pd.notna(c_index)
            else "Holdout concordance index: unavailable (no comparable pairs)"
        )
    except Exception as evaluation_error:
        evaluation = None
        evaluation_label = f"Holdout evaluation unavailable: {evaluation_error}"

    model = WeibullAFTModel().fit(
        training_data,
        training_data["duration_days"],
        training_data["event_observed"],
    )

    metrics = st.columns(4)
    metrics[0].metric("Training observations", f"{len(training_data):,}")
    metrics[1].metric("Paid events", f"{paid_count:,}")
    metrics[2].metric("Right-censored", f"{censored_count:,}")
    metrics[3].metric("Weibull shape", f"{model.shape_:.3f}")
    st.caption(
        "Model: Weibull accelerated-failure-time (AFT), fitted by maximum likelihood "
        "with right-censored observations. " + evaluation_label
    )
    if not model.converged_:
        st.warning(f"The optimizer did not report full convergence: {model.message_}")

    st.subheader("Payment-time survival curve")
    km_curve = kaplan_meier(
        training_data["duration_days"],
        training_data["event_observed"],
    )
    st.line_chart(km_curve.set_index("day")["survival_probability"])
    st.caption(
        "Kaplan-Meier is a non-parametric baseline. The Weibull AFT model adds "
        "invoice-level covariates for conditional forecasts."
    )

    st.subheader("Portfolio cash-flow simulation")
    call_events = load_json_list(CALL_EVENTS_PATH)
    portfolio = build_open_invoice_features(
        invoices,
        call_history,
        call_events,
        today=date.today(),
    )
    if portfolio.empty:
        st.info("No open invoices are available for simulation.")
    else:
        portfolio_total = float(portfolio["amount"].sum())
        c1, c2 = st.columns(2)
        with c1:
            collection_target = st.number_input(
                "Minimum collections target (₹)",
                min_value=0.0,
                value=float(round(portfolio_total * 0.5, 2)),
                step=1000.0,
                key="quant_collection_target",
            )
        with c2:
            simulation_count = st.select_slider(
                "Monte Carlo simulations",
                options=[1000, 2500, 5000, 10000, 20000],
                value=10000,
                key="quant_simulation_count",
            )

        st.caption(
            f"Open portfolio: ₹{portfolio_total:,.0f} across {len(portfolio)} invoices. "
            "Payment times are simulated conditionally on each invoice remaining unpaid "
            "until today."
        )
        summary, remaining_samples = simulate_portfolio_collections(
            model,
            portfolio,
            horizons=(7, 14, 30),
            n_simulations=int(simulation_count),
            minimum_collection_target=float(collection_target),
            confidence_level=0.95,
            random_seed=42,
        )

        display = summary.copy()
        currency_columns = [
            "expected_collections", "p05_collections", "median_collections",
            "p95_collections", "shortfall_var", "shortfall_expected_shortfall",
        ]
        for column in currency_columns:
            display[column] = display[column].map(lambda value: f"₹{value:,.0f}")
        display["probability_below_target"] = display["probability_below_target"].map(
            lambda value: f"{value:.1%}"
        )
        display = display.rename(columns={
            "horizon_days": "Horizon (days)",
            "expected_collections": "Expected collections",
            "p05_collections": "5th percentile",
            "median_collections": "Median collections",
            "p95_collections": "95th percentile",
            "probability_below_target": "P(collections below target)",
            "shortfall_var": "95% shortfall VaR",
            "shortfall_expected_shortfall": "95% shortfall ES",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption(
            "Shortfall loss = max(0, target − collections). VaR is the 95th percentile "
            "of this loss distribution; ES is the mean loss at or beyond that percentile. "
            "Baseline simulation assumes conditional independence between invoices."
        )

        st.subheader("Invoice-level payment estimates")
        invoice_estimates = portfolio.copy()
        invoice_estimates["estimated_median_remaining_days"] = model.predict_conditional_median_remaining(
            invoice_estimates,
            invoice_estimates["age_days"],
        )
        invoice_estimates["P(payment next 7d)"] = model.conditional_payment_probability(
            invoice_estimates, invoice_estimates["age_days"], 7
        )
        invoice_estimates["P(payment next 30d)"] = model.conditional_payment_probability(
            invoice_estimates, invoice_estimates["age_days"], 30
        )
        invoice_view = invoice_estimates[[
            "invoice_id", "customer", "amount", "age_days",
            "estimated_median_remaining_days", "P(payment next 7d)", "P(payment next 30d)",
        ]].rename(columns={
            "invoice_id": "Invoice",
            "customer": "Customer",
            "amount": "Outstanding (₹)",
            "age_days": "Days overdue",
            "estimated_median_remaining_days": "Estimated median days to payment",
            "P(payment next 7d)": "Payment probability (7d)",
            "P(payment next 30d)": "Payment probability (30d)",
        })
        invoice_view["Outstanding (₹)"] = invoice_view["Outstanding (₹)"].map(lambda value: f"₹{value:,.0f}")
        invoice_view["Estimated median days to payment"] = invoice_view["Estimated median days to payment"].map(lambda value: f"{value:.1f}")
        for column in ["Payment probability (7d)", "Payment probability (30d)"]:
            invoice_view[column] = invoice_view[column].map(lambda value: f"{value:.1%}")
        st.dataframe(invoice_view, use_container_width=True, hide_index=True)

    st.subheader("Call-outcome transition analysis")
    transition_matrix, transition_count = estimate_transition_matrix(call_events)
    if transition_count < 5:
        st.info(
            f"Only {transition_count} within-invoice status transitions are recorded. "
            "The smoothed Markov transition matrix is withheld until at least five "
            "transitions exist; continue retaining call events rather than overwriting history."
        )
    else:
        st.dataframe(
            transition_matrix.style.format("{:.1%}"),
            use_container_width=True,
        )

    st.subheader("Record a payment observation")
    st.caption(
        "Use one final record per invoice episode. A still-unpaid invoice is right-censored "
        "at the observation date; a paid invoice records an observed payment event. "
        "Only genuine, authorized business records belong in this file."
    )
    invoice_labels = {
        f'{item["invoice_id"]} · {item.get("customer", "Unknown")}': item
        for item in invoices
    }
    with st.form("record_payment_observation_form"):
        chosen_label = st.selectbox("Invoice", list(invoice_labels.keys()))
        chosen_invoice = invoice_labels[chosen_label]
        outcome = st.radio(
            "Observation outcome",
            ["Paid", "Still unpaid (right-censored)"],
            horizontal=True,
        )
        observation_date = st.date_input(
            "Payment date (if paid) / last observed date (if unpaid)",
            value=date.today(),
        )
        prior_delay = st.number_input("Customer's prior average delay (days)", min_value=0.0, value=float(chosen_invoice.get("prior_delay_days", 0)), step=1.0)
        existing_followups = sum(
            1 for event in call_events
            if event.get("invoice_id") == chosen_invoice["invoice_id"]
        )
        follow_up_count = st.number_input(
            "Number of follow-up calls to date",
            min_value=0,
            value=int(existing_followups),
            step=1,
        )
        latest_status = call_history.get(chosen_invoice["invoice_id"], {}).get("payment_status", "unknown")
        promised_flag = int(latest_status == "payment_promised")
        submitted = st.form_submit_button("Save payment observation")
        if submitted:
            due_date = datetime.strptime(chosen_invoice["due_date"], "%Y-%m-%d").date()
            duration = (observation_date - due_date).days
            if duration <= 0:
                st.error("The observation date must be after the invoice due date.")
            else:
                row = {
                    "invoice_id": chosen_invoice["invoice_id"],
                    "customer_id": chosen_invoice.get("customer_id", chosen_invoice.get("customer", "unknown")),
                    "amount": float(chosen_invoice["amount"]),
                    "prior_delay_days": float(prior_delay),
                    "follow_up_count": int(follow_up_count),
                    "promised_flag": promised_flag,
                    "duration_days": float(duration),
                    "event_observed": 1 if outcome == "Paid" else 0,
                    "observation_date": observation_date.isoformat(),
                    "data_source": "observed_user_entry",
                }
                try:
                    save_observation(OBSERVATIONS_PATH, row)
                    st.success(
                        f"Saved {outcome.lower()} observation for {chosen_invoice['invoice_id']}. "
                        "The model will switch to observed data once the minimum sample requirements are met."
                    )
                except Exception as save_error:
                    st.error(f"Could not save observation: {save_error}")

except Exception as quant_error:
    st.error(f"Quantitative analytics could not be loaded: {quant_error}")
    st.caption(
        "Check requirements.txt, data/demo_payment_observations.csv, and the "
        "observation CSV schema. The existing CALL-E follow-up workflow remains separate."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FinCall · AI-powered financial operations "
    "using CALL-E"
)