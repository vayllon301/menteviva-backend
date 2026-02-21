from typing import Optional
from dotenv import load_dotenv
import os
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "")
TWILIO_WHATSAPP_TO = os.getenv("TWILIO_WHATSAPP_TO", "")


def _ensure_whatsapp_prefix(number: str) -> str:
    if not number.startswith("whatsapp:"):
        return f"whatsapp:{number}"
    return number


def send_whatsapp_alert(message: str, to: Optional[str] = None) -> dict:
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        return {
            "error": "No se han configurado TWILIO_ACCOUNT_SID y/o TWILIO_AUTH_TOKEN",
            "alert": None
        }

    if not TWILIO_WHATSAPP_FROM:
        return {
            "error": "No se ha configurado TWILIO_WHATSAPP_FROM",
            "alert": None
        }

    recipient = to or TWILIO_WHATSAPP_TO
    if not recipient:
        return {
            "error": "No se ha proporcionado un número de destino ni se ha configurado TWILIO_WHATSAPP_TO",
            "alert": None
        }

    sender = _ensure_whatsapp_prefix(TWILIO_WHATSAPP_FROM)
    recipient = _ensure_whatsapp_prefix(recipient)

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        twilio_message = client.messages.create(
            from_=sender,
            body=message,
            to=recipient
        )

        return {
            "error": None,
            "alert": {
                "sid": twilio_message.sid,
                "estado": twilio_message.status,
                "destino": recipient,
                "mensaje": message
            }
        }

    except TwilioRestException as e:
        return {
            "error": f"Error de Twilio: {e.msg}",
            "alert": None
        }
    except Exception as e:
        return {
            "error": f"Error inesperado: {str(e)}",
            "alert": None
        }
