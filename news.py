from dotenv import load_dotenv
import os
import requests
from datetime import datetime, timedelta

load_dotenv()

# NewsAPI configuration
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_API_URL = "https://newsapi.org/v2/top-headlines"

def get_spain_news(limit: int = 10):
    """
    Obtiene noticias recientes de España desde NewsAPI.
    
    Args:
        limit: Número máximo de noticias a retornar (por defecto 10)
        
    Returns:
        Una lista de diccionarios con información de las noticias o un mensaje de error
    """
    if not NEWS_API_KEY:
        return {
            "error": "No se ha configurado NEWS_API_KEY en las variables de entorno",
            "news": []
        }
    
    try:
        # Parámetros para obtener noticias de España
        params = {
            "country": "es",
            "apiKey": NEWS_API_KEY,
            "pageSize": limit,
            "language": "es"
        }
        
        # Realizar la petición a NewsAPI
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("status") == "ok":
            articles = data.get("articles", [])
            
            # Formatear las noticias para una respuesta más limpia
            formatted_news = []
            for article in articles:
                formatted_news.append({
                    "titulo": article.get("title", "Sin título"),
                    "descripcion": article.get("description", ""),
                    "fuente": article.get("source", {}).get("name", "Fuente desconocida"),
                    "url": article.get("url", ""),
                    "fecha": article.get("publishedAt", ""),
                    "imagen": article.get("urlToImage", "")
                })
            
            return {
                "total": len(formatted_news),
                "news": formatted_news
            }
        else:
            return {
                "error": f"Error en la respuesta de NewsAPI: {data.get('message', 'Error desconocido')}",
                "news": []
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "error": f"Error al conectar con NewsAPI: {str(e)}",
            "news": []
        }
    except Exception as e:
        return {
            "error": f"Error inesperado: {str(e)}",
            "news": []
        }

def format_news_for_chat(news_data: dict) -> str:
    """
    Formatea las noticias en un formato legible para el chatbot.
    
    Args:
        news_data: Diccionario con las noticias obtenidas
        
    Returns:
        String formateado con las noticias
    """
    if news_data.get("error"):
        return f"Lo siento, no pude obtener las noticias. {news_data['error']}"
    
    news_list = news_data.get("news", [])
    
    if not news_list:
        return "No se encontraron noticias recientes de España en este momento."
    
    formatted_text = f"📰 Noticias recientes de España ({news_data.get('total', 0)} noticias):\n\n"
    
    for idx, news in enumerate(news_list, 1):
        formatted_text += f"{idx}. {news['titulo']}\n"
        if news.get('descripcion'):
            formatted_text += f"   {news['descripcion']}\n"
        formatted_text += f"   Fuente: {news['fuente']}\n"
        if news.get('fecha'):
            # Formatear la fecha de manera más legible
            try:
                date_obj = datetime.fromisoformat(news['fecha'].replace('Z', '+00:00'))
                formatted_date = date_obj.strftime("%d/%m/%Y %H:%M")
                formatted_text += f"   Fecha: {formatted_date}\n"
            except:
                formatted_text += f"   Fecha: {news['fecha']}\n"
        formatted_text += "\n"
    
    return formatted_text.strip()
