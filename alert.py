from typing import Optional
from dotenv import load_dotenv
import os
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_SMS_FROM = os.getenv("TWILIO_SMS_FROM", "")
TWILIO_WHATSAPP_TO = os.getenv("TWILIO_WHATSAPP_TO", "")


def _build_sms_body(
    user_name: Optional[str],
    latitude: Optional[float],
    longitude: Optional[float],
    description: Optional[str],
) -> str:
    lines = ["🚨 ALERTA DE EMERGENCIA - MenteViva"]
    lines.append("")

    if user_name:
        lines.append(f"👤 Enviada por: {user_name}")
    else:
        lines.append("👤 Enviada por: usuario desconocido")

    if latitude is not None and longitude is not None:
        lines.append(f"📍 Ubicación GPS: {latitude:.6f}, {longitude:.6f}")
        lines.append(f"🗺️ Ver en mapa: https://maps.google.com/?q={latitude},{longitude}")
    else:
        lines.append("📍 Ubicación: no disponible")

    if description and description.strip():
        lines.append("")
        lines.append(f"📝 Contexto: {description.strip()}")

    lines.append("")
    lines.append("Por favor, contacte con esta persona lo antes posible.")
    return "\n".join(lines)


def send_sms_alert(
    to: Optional[str] = None,
    user_name: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    description: Optional[str] = None,
) -> dict:
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        return {
            "error": "No se han configurado TWILIO_ACCOUNT_SID y/o TWILIO_AUTH_TOKEN",
            "alert": None,
        }

    if not TWILIO_SMS_FROM:
        return {
            "error": "No se ha configurado TWILIO_SMS_FROM",
            "alert": None,
        }

    recipient = to or TWILIO_WHATSAPP_TO
    if not recipient:
        return {
            "error": "No se ha proporcionado un número de destino",
            "alert": None,
        }

    # Strip any whatsapp: prefix -- we're sending SMS now
    recipient = recipient.replace("whatsapp:", "")

    body = _build_sms_body(user_name, latitude, longitude, description)

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        twilio_message = client.messages.create(
            from_=TWILIO_SMS_FROM,
            to=recipient,
            body=body,
        )

        return {
            "error": None,
            "alert": {
                "sid": twilio_message.sid,
                "estado": twilio_message.status,
                "destino": recipient,
            },
        }

    except TwilioRestException as e:
        return {
            "error": f"Error de Twilio: {e.msg}",
            "alert": None,
        }
    except Exception as e:
        return {
            "error": f"Error inesperado: {str(e)}",
            "alert": None,
        }
