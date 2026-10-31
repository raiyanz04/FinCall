"""Optional manual CALL-E smoke test. This script can place a real phone call.

Set FINCALL_TEST_PHONE in .env to a number you own or are authorized to call.
It is deliberately not named test_*.py so pytest will not execute it.
"""
import os

from dotenv import load_dotenv
from calle import CalleClient

load_dotenv()

api_key = os.getenv("CALLE_API_KEY")
phone_number = os.getenv("FINCALL_TEST_PHONE")

if not api_key:
    raise RuntimeError("CALLE_API_KEY not found in .env")
if not phone_number:
    raise RuntimeError("Set FINCALL_TEST_PHONE in .env to an authorized number")

print(f"This will place a real CALL-E test call to {phone_number}.")
confirmation = input('Type "I HAVE CONSENT" to proceed: ').strip()
if confirmation != "I HAVE CONSENT":
    raise SystemExit("Call cancelled.")

client = CalleClient(api_key=api_key)
result = client.calls.create_and_wait(
    task=(
        f"Call {phone_number}. This is a short technical test call for FinCall. "
        "Identify yourself as an AI test assistant, ask whether the recipient can hear you, "
        "thank them, and end the call politely. Do not request or disclose personal information."
    ),
    result_schema={
        "type": "object",
        "required": ["can_hear_clearly"],
        "properties": {
            "can_hear_clearly": {
                "type": "string",
                "enum": ["yes", "no", "unknown"],
            }
        },
    },
)

print("CALL-E result:")
print(result)
