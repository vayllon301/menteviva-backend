from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import ToolMessage
import pytest


def test_get_search_queries_uses_mini_model_for_profile_based_search(monkeypatch) -> None:
    import activities

    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='["biblioteca municipal", "museo local"]')
                    )
                ]
            )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )

    monkeypatch.setattr(activities, "openai_client", fake_client)

    queries = activities.get_search_queries(
        {
            "interests": "lectura y museos",
            "description": "Le gustan los planes tranquilos",
            "city": "Barcelona",
        },
        tutor_factors="movilidad reducida leve",
    )

    assert queries == ["biblioteca municipal", "museo local"]
    assert captured["model"] == "gpt-5.4-mini"
    assert "lectura y museos" in captured["messages"][0]["content"]
    assert "movilidad reducida leve" in captured["messages"][0]["content"]


def test_get_search_queries_uses_interest_specific_fallback_for_chess_profile(monkeypatch) -> None:
    import activities

    monkeypatch.setattr(activities, "openai_client", None)

    queries = activities.get_search_queries(
        {
            "interests": "ajedrez, juegos de mesa",
            "description": "Le encanta jugar al ajedrez y conocer gente",
            "city": "Barcelona",
        }
    )

    assert any("ajedrez" in query.lower() for query in queries)
    assert any("club" in query.lower() or "asociacion" in query.lower() for query in queries)


def test_search_activities_includes_search_summary(monkeypatch) -> None:
    import activities

    monkeypatch.setattr(activities, "GOOGLE_PLACES_API_KEY", "test-key")
    monkeypatch.setattr(
        activities,
        "get_search_plan",
        lambda user_profile, tutor_factors="": {
            "summary": "He buscado clubes de ajedrez y espacios tranquilos para jugar partidas.",
            "queries": ["club de ajedrez", "asociacion de ajedrez"],
        },
    )
    monkeypatch.setattr(
        activities,
        "search_places",
        lambda queries, lat, lng, radius_m=10000: [
            {
                "name": "Club d'Escacs Barcelona",
                "address": "C/ Major 1",
                "rating": 4.7,
                "open_now": True,
                "types": ["point_of_interest"],
                "place_id": "chess-1",
            }
        ],
    )
    monkeypatch.setattr(
        activities,
        "personalize_results",
        lambda places, user_profile, tutor_factors="": [
            {**places[0], "recommendation": "Ideal para jugar partidas y conocer gente."}
        ],
    )

    result = activities.search_activities(
        user_profile={"interests": "ajedrez", "description": "Le encanta jugar", "city": "Barcelona"},
        latitude=41.3874,
        longitude=2.1686,
    )

    assert "He buscado clubes de ajedrez" in result
    assert "Club d'Escacs Barcelona" in result


def test_search_activities_requires_location_or_city(monkeypatch) -> None:
    import activities

    monkeypatch.setattr(activities, "GOOGLE_PLACES_API_KEY", "test-key")

    result = activities.search_activities(user_profile={})

    assert "No conozco tu ubicación" in result


def test_search_activities_geocodes_profile_city_when_needed(monkeypatch) -> None:
    import activities

    monkeypatch.setattr(activities, "GOOGLE_PLACES_API_KEY", "test-key")

    captured: dict = {}

    def fake_geocode_city(city: str) -> dict:
        captured["city"] = city
        return {"lat": 41.3874, "lng": 2.1686}

    monkeypatch.setattr(activities, "geocode_city", fake_geocode_city)
    monkeypatch.setattr(
        activities,
        "get_search_queries",
        lambda user_profile, tutor_factors="": ["biblioteca municipal"],
    )
    monkeypatch.setattr(
        activities,
        "search_places",
        lambda queries, lat, lng, radius_m=10000: [
            {
                "name": "Biblioteca Municipal",
                "address": "C/ Mayor 1",
                "rating": 4.6,
                "open_now": True,
                "types": ["library"],
                "place_id": "abc123",
            }
        ],
    )
    monkeypatch.setattr(
        activities,
        "personalize_results",
        lambda places, user_profile, tutor_factors="": [
            {**places[0], "recommendation": "Ideal para leer con calma."}
        ],
    )

    result = activities.search_activities(user_profile={"city": "Barcelona"})

    assert captured["city"] == "Barcelona"
    assert "Biblioteca Municipal" in result
    assert "Ideal para leer con calma." in result


def test_chatbot_passes_user_location_to_graph(monkeypatch) -> None:
    import chatbot

    captured: dict = {}

    class FakeGraph:
        def invoke(self, input_state: dict) -> dict:
            captured["state"] = input_state
            return {"messages": [SimpleNamespace(content="respuesta de prueba")]}

    monkeypatch.setattr(chatbot, "graph", FakeGraph())

    result = chatbot.chatbot(
        "hola",
        user_location={"latitude": 41.3874, "longitude": 2.1686},
    )

    assert result == "respuesta de prueba"
    assert captured["state"]["user_location"] == {
        "latitude": 41.3874,
        "longitude": 2.1686,
    }


def test_chatbot_node_preserves_activity_tool_output() -> None:
    import chatbot

    tool_output = (
        "1. Biblioteca Municipal - C/ Mayor 1\n"
        "   ⭐ 4.6 | Abierto ahora\n"
        "   -> Ideal para leer con calma."
    )

    result = chatbot.chatbot_node(
        {
            "messages": [
                ToolMessage(
                    content=tool_output,
                    tool_call_id="call_test",
                    name="buscar_actividades",
                )
            ],
            "user_profile": {},
            "tutor_profile": {},
            "user_memory": {},
            "user_location": {},
        }
    )

    assert result["messages"][0].content == tool_output


@pytest.mark.asyncio
async def test_chatbot_stream_falls_back_to_chain_end_output(monkeypatch) -> None:
    import chatbot

    async def fake_astream_events(input_state: dict, version: str = "v2"):
        yield {
            "event": "on_chain_end",
            "name": "chatbot",
            "data": {
                "output": {
                    "messages": [SimpleNamespace(content="1. Biblioteca Municipal - C/ Mayor 1")]
                }
            },
        }

    monkeypatch.setattr(chatbot.graph, "astream_events", fake_astream_events)

    chunks = []
    async for chunk in chatbot.chatbot_stream("hola"):
        chunks.append(chunk)

    assert chunks == ["1. Biblioteca Municipal - C/ Mayor 1"]


def test_chat_endpoint_passes_coordinates_to_chatbot(monkeypatch) -> None:
    import main

    captured: dict = {}

    async def fake_scheduler_loop() -> None:
        return None

    async def fake_chatbot_async(
        message: str,
        history: list | None = None,
        user_profile: dict | None = None,
        tutor_profile: dict | None = None,
        user_memory: dict | None = None,
        user_location: dict | None = None,
    ) -> str:
        captured["message"] = message
        captured["user_location"] = user_location
        return "ok"

    monkeypatch.setattr(main, "scheduler_loop", fake_scheduler_loop)
    monkeypatch.setattr(main, "chatbot_async", fake_chatbot_async)

    with TestClient(main.app) as client:
        response = client.post(
            "/chat",
            json={
                "message": "hola",
                "history": [],
                "latitude": 41.3874,
                "longitude": 2.1686,
            },
        )

    assert response.status_code == 200
    assert response.json() == {"response": "ok"}
    assert captured["message"] == "hola"
    assert captured["user_location"] == {
        "latitude": 41.3874,
        "longitude": 2.1686,
    }


def test_chat_stream_endpoint_passes_coordinates_to_chatbot(monkeypatch) -> None:
    import main

    captured: dict = {}

    async def fake_scheduler_loop() -> None:
        return None

    async def fake_chatbot_stream(
        message: str,
        history: list | None = None,
        user_profile: dict | None = None,
        tutor_profile: dict | None = None,
        user_memory: dict | None = None,
        user_location: dict | None = None,
    ):
        captured["message"] = message
        captured["user_location"] = user_location
        yield "hola"

    monkeypatch.setattr(main, "scheduler_loop", fake_scheduler_loop)
    monkeypatch.setattr(main, "chatbot_stream", fake_chatbot_stream)

    with TestClient(main.app) as client:
        response = client.post(
            "/chat/stream",
            json={
                "message": "hola",
                "history": [],
                "latitude": 41.3874,
                "longitude": 2.1686,
            },
        )

    assert response.status_code == 200
    assert "data:" in response.text
    assert captured["message"] == "hola"
    assert captured["user_location"] == {
        "latitude": 41.3874,
        "longitude": 2.1686,
    }
