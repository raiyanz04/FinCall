# FinCall

AI-powered financial operations phone agent built with CALL-E.

## Problem

Finance teams spend significant time following up with customers
about overdue invoices. These calls are repetitive, time-consuming,
and difficult to track consistently.

## Solution

FinCall prioritizes overdue invoices and uses CALL-E to make an
AI-powered phone call to the customer.

The agent:

1. Selects an overdue invoice.
2. Calls the customer through CALL-E.
3. Discusses the outstanding payment.
4. Determines the customer's payment status.
5. Extracts the expected payment date.
6. Determines whether follow-up is required.
7. Recommends the next action.

## Features

- Overdue invoice prioritization
- AI-powered outbound phone calls
- Structured payment-status extraction
- Expected payment-date extraction
- Call evidence
- Recommended next actions
- Streamlit dashboard
- CALL-E Python SDK integration

## Architecture

Streamlit UI
    ↓
FinCall Python application
    ↓
CALL-E Python SDK
    ↓
AI phone conversation
    ↓
Structured financial outcome
    ↓
Recommended next action

## Tech Stack

- Python
- Streamlit
- CALL-E
- Python-dotenv

## Setup

```bash
git clone <YOUR_REPOSITORY_URL>
cd FinCall

python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
