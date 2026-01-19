from dotenv import load_dotenv
import os
import requests
from datetime import datetime

load_dotenv()

# WeatherAPI.com API configuration
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"

def get_weather(city: str = "Madrid", country_code: str = "ES"):
    """
    Obtiene el clima actual de una ciudad usando WeatherAPI.com.
    
    Args:
        city: Nombre de la ciudad (por defecto "Madrid")
        country_code: Código del país (por defecto "ES" para España)
        
    Returns:
        Un diccionario con información del clima o un mensaje de error
    """
    if not WEATHER_API_KEY:
        return {
            "error": "No se ha configurado WEATHER_API_KEY en las variables de entorno",
            "weather": None
        }
    
    try:
        # Parámetros para obtener el clima
        params = {
            "key": WEATHER_API_KEY,
            "q": f"{city},{country_code}",
            "lang": "es",  # Respuestas en español
            "aqi": "no"  # No incluir calidad del aire
        }
        
        # Realizar la petición a WeatherAPI
        response = requests.get(WEATHER_API_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Formatear la respuesta según la estructura de WeatherAPI
        location = data.get("location", {})
        current = data.get("current", {})
        condition = current.get("condition", {})
        
        weather_info = {
            "ciudad": location.get("name", city),
            "pais": location.get("country", country_code),
            "temperatura": round(current.get("temp_c", 0)),
            "sensacion_termica": round(current.get("feelslike_c", 0)),
            "descripcion": condition.get("text", "").capitalize(),
            "humedad": current.get("humidity", 0),
            "presion": round(current.get("pressure_mb", 0)),
            "viento_velocidad": round(current.get("wind_kph", 0), 1),
            "fecha_actualizacion": location.get("localtime", None)
        }
        
        return {
            "error": None,
            "weather": weather_info
        }
            
    except requests.exceptions.HTTPError as e:
        if response.status_code == 400:
            error_data = response.json() if response.text else {}
            error_message = error_data.get("error", {}).get("message", "Solicitud inválida")
            return {
                "error": f"No se encontró la ciudad '{city}'. {error_message}",
                "weather": None
            }
        elif response.status_code == 401 or response.status_code == 403:
            return {
                "error": "API key inválida. Por favor, verifica WEATHER_API_KEY en las variables de entorno.",
                "weather": None
            }
        else:
            return {
                "error": f"Error HTTP al conectar con WeatherAPI: {str(e)}",
                "weather": None
            }
    except requests.exceptions.RequestException as e:
        return {
            "error": f"Error al conectar con WeatherAPI: {str(e)}",
            "weather": None
        }
    except Exception as e:
        return {
            "error": f"Error inesperado: {str(e)}",
            "weather": None
        }

def format_weather_for_chat(weather_data: dict) -> str:
    """
    Formatea la información del clima para que el chatbot lo interprete y lo presente de forma natural.
    
    Args:
        weather_data: Diccionario con la información del clima obtenida
        
    Returns:
        String con datos formateados para que el chatbot lo presente de forma natural
    """
    if weather_data.get("error"):
        return f"Error al obtener el clima: {weather_data['error']}"
    
    weather = weather_data.get("weather")
    if not weather:
        return "No se pudo obtener la información del clima."
    
    # Crear un resumen conciso de los datos para que el chatbot lo interprete
    resumen = (
        f"Ciudad: {weather['ciudad']}, {weather['pais']}\n"
        f"Temperatura: {weather['temperatura']}°C (sensación térmica: {weather['sensacion_termica']}°C)\n"
        f"Condiciones: {weather['descripcion']}\n"
        f"Humedad: {weather['humedad']}%\n"
        f"Viento: {weather['viento_velocidad']} km/h"
    )
    
    
    
    if weather.get('fecha_actualizacion'):
        resumen += f"\nÚltima actualización: {weather['fecha_actualizacion']}"
    
    return resumen
