from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
import unicodedata

load_dotenv()

GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
GEOCODING_URL = "https://maps.googleapis.com/maps/api/geocode/json"
SEARCH_QUERY_MODEL = "gpt-5.4-mini"
RESULT_PERSONALIZATION_MODEL = "gpt-5.4-nano"

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

INTEREST_QUERY_RULES = [
    (
        ("ajedrez", "escacs"),
        {
            "summary": "He buscado clubes de ajedrez y espacios tranquilos para jugar partidas y socializar.",
            "queries": [
                "club de ajedrez",
                "asociacion de ajedrez",
                "centro civico ajedrez",
            ],
        },
    ),
    (
        ("lectura", "libros", "leer"),
        {
            "summary": "He buscado bibliotecas y espacios tranquilos con actividades de lectura.",
            "queries": [
                "biblioteca municipal",
                "club de lectura",
                "centro cultural lectura",
            ],
        },
    ),
    (
        ("museo", "arte", "pintura"),
        {
            "summary": "He buscado museos y centros culturales con planes tranquilos e interesantes.",
            "queries": [
                "museo",
                "centro cultural",
                "sala de exposiciones",
            ],
        },
    ),
    (
        ("musica", "cantar", "coro"),
        {
            "summary": "He buscado actividades musicales y espacios donde disfrutar de la musica con calma.",
            "queries": [
                "escuela de musica",
                "centro civico musica",
                "coro local",
            ],
        },
    ),
]


def _extract_json_content(content: str) -> str:
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def _keyword_fallback_search_plan(user_profile: dict) -> dict:
    interests = user_profile.get("interests", "") or ""
    description = user_profile.get("description", "") or ""
    profile_text = _normalize_text(f"{interests} {description}")

    for keywords, plan in INTEREST_QUERY_RULES:
        if any(keyword in profile_text for keyword in keywords):
            return {
                "summary": plan["summary"],
                "queries": plan["queries"][:5],
            }

    return {
        "summary": "He buscado actividades tranquilas y sociales que encajen con el perfil del usuario.",
        "queries": ["centro de mayores", "biblioteca", "centro cívico"],
    }


GENERAL_ELDERLY_QUERIES = [
    "parque público",
    "centro de mayores",
    "paseo bonito",
    "plaza del pueblo",
    "jardín botánico",
    "centro cívico",
    "cafetería tranquila",
    "mercado municipal",
]


def get_general_search_plan(user_profile: dict) -> dict:
    """
    Build 2-3 generic, age-appropriate Google Places queries
    based on the user's location (not their specific hobbies).
    These are places any elderly person might enjoy.
    """
    city = user_profile.get("city", "") or ""
    description = user_profile.get("description", "") or ""

    prompt = (
        "Eres un asistente que prepara búsquedas en Google Places para personas mayores.\n"
        f"Ciudad del usuario: {city}\n"
        f"Descripción del usuario: {description}\n\n"
        "Genera 2 o 3 búsquedas GENERALES de lugares que cualquier persona mayor "
        "disfrutaría, independientemente de sus aficiones concretas. "
        "Piensa en: parques tranquilos, plazas bonitas, centros de mayores, "
        "cafeterías acogedoras, mercados, jardines, paseos...\n"
        "NO incluyas actividades específicas de ningún hobby.\n"
        "Responde SOLO con un JSON object con este formato:\n"
        '{"summary":"Breve resumen","queries":["query 1","query 2"]}\n'
        "Las queries deben ser útiles para Google Places y tener entre 2 y 3 elementos."
    )

    llm_output = _llm_json(
        prompt,
        temperature=0.3,
        max_tokens=150,
        model=SEARCH_QUERY_MODEL,
    )

    if isinstance(llm_output, dict):
        queries = llm_output.get("queries", [])
        summary = llm_output.get("summary", "")
        cleaned = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        if cleaned:
            return {
                "summary": summary.strip() or "También he buscado lugares agradables por tu zona.",
                "queries": cleaned[:3],
            }

    # Fallback: pick 2 generic queries
    return {
        "summary": "También he buscado lugares agradables por tu zona.",
        "queries": GENERAL_ELDERLY_QUERIES[:2],
    }


def _build_search_summary_from_queries(queries: list[str]) -> str:
    if not queries:
        return "He buscado actividades interesantes cerca de ti."
    preview = ", ".join(queries[:2])
    return f"He buscado opciones como {preview} cerca de ti."


def get_search_plan(user_profile: dict, tutor_factors: str = "") -> dict:
    """
    Build a brief search summary and 3 SPECIFIC Google Places queries
    based on the user's interests/hobbies.
    """
    interests = user_profile.get("interests", "") or ""
    description = user_profile.get("description", "") or ""
    city = user_profile.get("city", "") or ""

    prompt = (
        "Eres un asistente que prepara busquedas en Google Places para personas mayores.\n"
        f"Perfil del usuario:\n"
        f"- Intereses: {interests}\n"
        f"- Descripcion: {description}\n"
        f"- Ciudad: {city}\n"
    )
    if tutor_factors:
        prompt += f"- Condiciones de salud/factores importantes: {tutor_factors}\n"
    prompt += (
        "\nPiensa en actividades o lugares MUY concretos que encajen con sus intereses. "
        "Si le gusta el ajedrez, busca cosas como clubes o asociaciones de ajedrez. "
        "Si le gusta leer, piensa en bibliotecas o clubes de lectura. "
        "Evita terminos genericos si hay una aficion clara.\n"
        "Responde SOLO con un JSON object con este formato:\n"
        '{"summary":"Breve resumen de la busqueda","queries":["query 1","query 2","query 3"]}\n'
        "Las queries deben ser utiles para Google Places y tener EXACTAMENTE 3 elementos, "
        "todos muy específicos a los intereses del usuario."
    )

    llm_output = _llm_json(
        prompt,
        temperature=0.2,
        max_tokens=220,
        model=SEARCH_QUERY_MODEL,
    )

    if isinstance(llm_output, dict):
        queries = llm_output.get("queries", [])
        summary = llm_output.get("summary", "")
        cleaned_queries = [
            query.strip()
            for query in queries
            if isinstance(query, str) and query.strip()
        ]
        if cleaned_queries:
            return {
                "summary": summary.strip() or _build_search_summary_from_queries(cleaned_queries),
                "queries": cleaned_queries[:5],
            }

    if isinstance(llm_output, list):
        cleaned_queries = [
            query.strip()
            for query in llm_output
            if isinstance(query, str) and query.strip()
        ]
        if cleaned_queries:
            fallback_plan = _keyword_fallback_search_plan(user_profile)
            return {
                "summary": fallback_plan["summary"] if any(
                    keyword in _normalize_text(" ".join(cleaned_queries))
                    for keywords, _ in INTEREST_QUERY_RULES
                    for keyword in keywords
                ) else _build_search_summary_from_queries(cleaned_queries),
                "queries": cleaned_queries[:5],
            }

    return _keyword_fallback_search_plan(user_profile)


def _llm_json(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 250,
    model: str = RESULT_PERSONALIZATION_MODEL,
):
    if not openai_client:
        return None

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        return json.loads(_extract_json_content(content))
    except Exception:
        return None


def geocode_city(city: str) -> dict:
    """
    Convert a city name to lat/lng using Google Geocoding API.

    Returns:
        {"lat": float, "lng": float} or {"error": str}
    """
    if not GOOGLE_PLACES_API_KEY:
        return {"error": "No se ha configurado GOOGLE_PLACES_API_KEY."}

    try:
        response = requests.get(
            GEOCODING_URL,
            params={
                "address": f"{city}, Spain",
                "key": GOOGLE_PLACES_API_KEY,
                "language": "es",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            return {"error": f"No se encontró la ciudad '{city}'."}
        location = results[0]["geometry"]["location"]
        return {"lat": location["lat"], "lng": location["lng"]}
    except requests.exceptions.RequestException as exc:
        return {"error": f"Error al geocodificar la ciudad: {str(exc)}"}


def get_search_queries(user_profile: dict, tutor_factors: str = "") -> list[str]:
    """
    Use a small LLM plus profile-based fallback to generate Google Places queries.
    """
    return get_search_plan(user_profile, tutor_factors).get("queries", [])


def search_places(queries: list[str], lat: float, lng: float, radius_m: int = 10000) -> list[dict]:
    """
    Search Google Places Nearby for each query and deduplicate by place_id.
    """
    if not GOOGLE_PLACES_API_KEY:
        return []

    seen_ids = set()
    places = []

    for query in queries:
        try:
            response = requests.get(
                PLACES_NEARBY_URL,
                params={
                    "location": f"{lat},{lng}",
                    "radius": radius_m,
                    "keyword": query,
                    "language": "es",
                    "key": GOOGLE_PLACES_API_KEY,
                },
                timeout=10,
            )
            response.raise_for_status()
            results = response.json().get("results", [])

            for place in results:
                place_id = place.get("place_id")
                if not place_id or place_id in seen_ids:
                    continue

                seen_ids.add(place_id)
                places.append(
                    {
                        "name": place.get("name", ""),
                        "address": place.get("vicinity", ""),
                        "rating": place.get("rating"),
                        "open_now": place.get("opening_hours", {}).get("open_now"),
                        "types": place.get("types", []),
                        "place_id": place_id,
                    }
                )
        except requests.exceptions.RequestException:
            continue

    return places


def personalize_results(
    places: list[dict],
    user_profile: dict,
    tutor_factors: str = "",
    max_results: int = 5,
    context_hint: str = "",
) -> list[dict]:
    """
    Use gpt-5.4-nano to rank places and add a warm recommendation.

    Args:
        max_results: How many results to return (default 5).
        context_hint: Extra instruction for the LLM about the type of results
                      (e.g. "general" vs "specific to interests").
    """
    if not places:
        return []

    interests = user_profile.get("interests", "") or ""
    description = user_profile.get("description", "") or ""

    places_text = "\n".join(
        (
            f"- {place['name']} ({place['address']}) - rating: {place.get('rating', 'N/A')}, "
            f"abierto: {'si' if place.get('open_now') else 'no' if place.get('open_now') is False else 'desconocido'}, "
            f"tipos: {', '.join(place.get('types', [])[:3])}"
        )
        for place in places[:20]
    )

    prompt = (
        "Eres un asistente calido que ayuda a personas mayores.\n"
        f"Perfil del usuario:\n"
        f"- Intereses: {interests}\n"
        f"- Descripcion: {description}\n"
    )
    if tutor_factors:
        prompt += f"- Condiciones de salud: {tutor_factors}\n"
    if context_hint:
        prompt += f"- Contexto: {context_hint}\n"
    prompt += (
        f"\nLugares encontrados cerca del usuario:\n{places_text}\n\n"
        f"Selecciona los {max_results} lugares mas apropiados para esta persona. "
        "Para cada uno, escribe una frase corta y calida explicando por que le puede interesar. "
        "Responde SOLO con un JSON array de objetos con los campos "
        '"name" y "recommendation". Ejemplo: '
        '[{"name": "Biblioteca Municipal", "recommendation": "Tienen un club de lectura los martes."}]'
    )

    recommendations = _llm_json(
        prompt,
        temperature=0.5,
        max_tokens=500,
        model=RESULT_PERSONALIZATION_MODEL,
    )
    if isinstance(recommendations, list):
        recommendation_map = {}
        for recommendation in recommendations:
            if (
                isinstance(recommendation, dict)
                and isinstance(recommendation.get("name"), str)
                and isinstance(recommendation.get("recommendation"), str)
            ):
                recommendation_map[recommendation["name"]] = recommendation["recommendation"].strip()

        personalized = []
        for place in places:
            if place["name"] in recommendation_map:
                personalized.append(
                    {
                        **place,
                        "recommendation": recommendation_map[place["name"]],
                    }
                )
        if personalized:
            return personalized[:max_results]

    return [
        {**place, "recommendation": "Un lugar interesante cerca de ti."}
        for place in places[:max_results]
    ]


def _format_activity_list(activities: list[dict], start_index: int = 1) -> list[str]:
    """Format a list of activities into lines, starting at the given index."""
    lines = []
    for i, activity in enumerate(activities, start_index):
        rating_text = f"⭐ {activity['rating']}" if activity.get("rating") else ""
        open_text = (
            "Abierto ahora"
            if activity.get("open_now")
            else "Cerrado ahora"
            if activity.get("open_now") is False
            else ""
        )
        status = " | ".join(part for part in [rating_text, open_text] if part)

        lines.append(f"{i}. {activity['name']} - {activity['address']}")
        if status:
            lines.append(f"   {status}")
        lines.append(f"   -> {activity.get('recommendation', '')}")
        lines.append("")
    return lines


def format_activities_for_chat(
    specific_activities: list[dict],
    general_activities: list[dict],
    specific_summary: str = "",
    general_summary: str = "",
) -> str:
    """Format both specific and general activities for the chatbot."""
    if not specific_activities and not general_activities:
        return "No se encontraron actividades cercanas."

    lines = []

    # Section 1: Specific interest-based results
    if specific_activities:
        if specific_summary.strip():
            lines.append(f"🎯 {specific_summary.strip()}")
            lines.append("")
        lines.extend(_format_activity_list(specific_activities, start_index=1))

    # Section 2: General age-appropriate results
    if general_activities:
        if specific_activities:
            lines.append("---")
            lines.append("")
        summary = general_summary.strip() or "También podrían interesarte estos lugares por tu zona:"
        lines.append(f"🌿 {summary}")
        lines.append("")
        start = len(specific_activities) + 1
        lines.extend(_format_activity_list(general_activities, start_index=start))

    return "\n".join(lines).strip()


def search_activities(
    user_profile: dict,
    tutor_factors: str = "",
    latitude: float = None,
    longitude: float = None,
    radius_km: int = 10,
) -> str:
    """
    Search for personalized activities near the user.

    Returns 3 specific (interest-based) + 2 general (age/location-based)
    recommendations.
    """
    if not GOOGLE_PLACES_API_KEY:
        return "Error: no se ha configurado la clave de Google Places."

    lat = latitude
    lng = longitude

    if lat is None or lng is None:
        city = (user_profile or {}).get("city", "")
        if not city:
            return (
                "No conozco tu ubicación. Puedes decirme tu ciudad "
                "o activar la ubicación en tu navegador."
            )

        geocoded = geocode_city(city)
        if "error" in geocoded:
            return geocoded["error"]
        lat = geocoded["lat"]
        lng = geocoded["lng"]

    radius_m = radius_km * 1000
    profile = user_profile or {}

    # --- 1. Specific interest-based search (3 results) ---
    specific_plan = get_search_plan(profile, tutor_factors)
    specific_queries = specific_plan.get("queries", [])
    specific_places = search_places(specific_queries, lat, lng, radius_m=radius_m)
    specific_personalized = personalize_results(
        specific_places,
        profile,
        tutor_factors,
        max_results=3,
        context_hint="Estos resultados son específicos para los intereses del usuario.",
    )

    # Collect place_ids already used so we don't repeat them in general results
    used_ids = {p["place_id"] for p in specific_personalized if p.get("place_id")}

    # --- 2. General age-appropriate search (2 results) ---
    general_plan = get_general_search_plan(profile)
    general_queries = general_plan.get("queries", [])
    general_places_raw = search_places(general_queries, lat, lng, radius_m=radius_m)
    # Remove duplicates that already appeared in specific results
    general_places = [p for p in general_places_raw if p.get("place_id") not in used_ids]
    general_personalized = personalize_results(
        general_places,
        profile,
        tutor_factors,
        max_results=2,
        context_hint=(
            "Estos son lugares generales que cualquier persona mayor disfrutaría. "
            "Escribe recomendaciones cálidas sin asumir aficiones concretas."
        ),
    )

    if not specific_personalized and not general_personalized:
        return (
            "No he encontrado actividades cercanas con ese criterio. "
            "¿Quieres que busque en un radio más amplio?"
        )

    return format_activities_for_chat(
        specific_activities=specific_personalized,
        general_activities=general_personalized,
        specific_summary=specific_plan.get("summary", ""),
        general_summary=general_plan.get("summary", ""),
    )
