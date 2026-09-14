import os
from dotenv import load_dotenv
from calle import CalleClient

load_dotenv()

api_key = os.getenv("CALLE_API_KEY")

if not api_key:
    raise RuntimeError("CALLE_API_KEY not found in .env")

client = CalleClient(api_key=api_key)

PHONE_NUMBER = "+917894040894"

call = client.calls.create_and_wait(
    task=f"""
    Call {PHONE_NUMBER}.

    This is a short technical test call for a hackathon project
    called FinCall.

    Say:
    "Hello, this is a test call from FinCall.
    We are testing an AI phone assistant.
    Can you hear me clearly?"

    Wait for the person to respond.

    If they say yes, thank them and end the call.
    If they say no or cannot hear clearly, acknowledge that
    and end the call.
    """,
    result_schema={
        "type": "object",
        "required": ["can_hear_clearly"],
        "properties": {
            "can_hear_clearly": {
                "type": "string",
                "enum": ["yes", "no", "unknown"]
            }
        }
    }
)

print("\n========== CALL RESULT ==========")
print("Status:", call["status"])
print("Task completed:", call["task_completed"])
print("Structured result:", call["structured_result"])
print("Evidence:", call["evidence"])
print("=================================\n")