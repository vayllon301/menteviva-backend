"""
Módulo para obtener noticias de periódicos españoles (El País y El Mundo)
usando sus RSS feeds.
"""
import feedparser
from datetime import datetime
from typing import List, Dict
from bs4 import BeautifulSoup

# URLs de los RSS feeds
ELPAIS_RSS = "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada"
ELMUNDO_RSS = "https://e00-elmundo.uecdn.es/elmundo/rss/portada.xml"

RSS_SOURCES = {
    "elpais": {"url": ELPAIS_RSS, "name": "El País"},
    "elmundo": {"url": ELMUNDO_RSS, "name": "El Mundo"},
}

def _get_rss_news(rss_url: str, source_name: str, limit: int = 5) -> List[Dict]:
    """Obtiene noticias de un RSS feed."""
    try:
        feed = feedparser.parse(rss_url)
        return [
            {
                "titulo": entry.get("title", "Sin título"),
                "descripcion": entry.get("summary", ""),
                "url": entry.get("link", ""),
                "fecha": entry.get("published", ""),
                "fuente": source_name,
            }
            for entry in feed.entries[:limit]
        ]
    except Exception as e:
        print(f"Error obteniendo noticias de {source_name}: {str(e)}")
        return []

def get_elpais_news(limit: int = 5) -> List[Dict]:
    return _get_rss_news(ELPAIS_RSS, "El País", limit)

def get_elmundo_news(limit: int = 5) -> List[Dict]:
    return _get_rss_news(ELMUNDO_RSS, "El Mundo", limit)

def get_combined_news(limit_per_source: int = 5) -> Dict:
    """
    Obtiene noticias combinadas de El País y El Mundo.
    
    Args:
        limit_per_source: Número de noticias a obtener de cada fuente
        
    Returns:
        Diccionario con las noticias de ambos periódicos
    """
    elpais_news = get_elpais_news(limit=limit_per_source)
    elmundo_news = get_elmundo_news(limit=limit_per_source)
    
    all_news = elpais_news + elmundo_news
    
    # Ordenar por fecha si es posible
    try:
        all_news.sort(
            key=lambda x: datetime.strptime(x['fecha'], '%a, %d %b %Y %H:%M:%S %z') if x['fecha'] else datetime.min,
            reverse=True
        )
    except (ValueError, TypeError):
        # Si hay error al parsear fechas, mantener el orden original
        pass
    
    return {
        "total": len(all_news),
        "fecha_consulta": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "news": all_news
    }

def format_newspapers_for_chat(news_data: Dict) -> str:
    """
    Formatea las noticias de los periódicos en un formato legible para el chatbot.
    
    Args:
        news_data: Diccionario con las noticias obtenidas
        
    Returns:
        String formateado con las noticias
    """
    if news_data.get("error"):
        return f"Lo siento, no pude obtener las noticias. {news_data['error']}"
    
    news_list = news_data.get("news", [])
    
    if not news_list:
        return "No se encontraron noticias recientes en este momento."
    
    # Incluir la fecha actual para contexto
    today = datetime.now().strftime("%A %d de %B de %Y")
    days_es = {
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
    }
    months_es = {
        "January": "Enero", "February": "Febrero", "March": "Marzo", "April": "Abril",
        "May": "Mayo", "June": "Junio", "July": "Julio", "August": "Agosto",
        "September": "Septiembre", "October": "Octubre", "November": "Noviembre", "December": "Diciembre"
    }
    
    # Traducir día y mes al español
    for eng, esp in days_es.items():
        today = today.replace(eng, esp)
    for eng, esp in months_es.items():
        today = today.replace(eng, esp)
    
    formatted_text = f"📰 Noticias de El País y El Mundo\n"
    formatted_text += f"📅 Fecha: {today}\n"
    formatted_text += f"Total de noticias: {news_data.get('total', 0)}\n\n"
    
    for idx, news in enumerate(news_list, 1):
        formatted_text += f"{idx}. [{news['fuente']}] {news['titulo']}\n"
        if news.get('descripcion'):
            # Limpiar HTML tags si existen
            desc = news['descripcion']
            if '<' in desc:
                desc = BeautifulSoup(desc, 'html.parser').get_text()
            # Limitar descripción si es muy larga
            if len(desc) > 200:
                desc = desc[:200] + "..."
            formatted_text += f"   {desc}\n"
        
        if news.get('fecha'):
            # Intentar formatear la fecha de manera más legible
            try:
                # Los RSS suelen usar formato RFC 2822
                date_obj = datetime.strptime(news['fecha'], '%a, %d %b %Y %H:%M:%S %z')
                formatted_date = date_obj.strftime("%d/%m/%Y %H:%M")
                formatted_text += f"   📅 {formatted_date}\n"
            except (ValueError, TypeError):
                formatted_text += f"   📅 {news['fecha']}\n"
        
        formatted_text += f"   🔗 {news['url']}\n\n"
    
    return formatted_text.strip()

def get_newspapers_by_source(source: str, limit: int = 5) -> Dict:
    """
    Obtiene noticias de un periódico específico.
    
    Args:
        source: Nombre del periódico ("elpais" o "elmundo")
        limit: Número de noticias a obtener
        
    Returns:
        Diccionario con las noticias del periódico solicitado
    """
    today = datetime.now().strftime("%d/%m/%Y %H:%M")
    
    if source.lower() == "elpais":
        news = get_elpais_news(limit=limit)
    elif source.lower() == "elmundo":
        news = get_elmundo_news(limit=limit)
    else:
        return {
            "error": f"Fuente no reconocida: {source}. Use 'elpais' o 'elmundo'",
            "news": []
        }
    
    return {
        "total": len(news),
        "fecha_consulta": today,
        "news": news
    }
