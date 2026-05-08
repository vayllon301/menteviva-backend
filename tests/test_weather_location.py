import pytest


def test_get_weather_queries_by_coordinates_when_available(monkeypatch) -> None:
    import weather

    captured: dict = {}

    class FakeResponse:
        status_code = 200
        text = "{}"

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "location": {
                    "name": "Barcelona",
                    "country": "Spain",
                    "localtime": "2026-05-08 12:00",
                },
                "current": {
                    "temp_c": 22.2,
                    "feelslike_c": 23.1,
                    "condition": {"text": "soleado"},
                    "humidity": 40,
                    "pressure_mb": 1015,
                    "wind_kph": 11.3,
                },
            }

    def fake_get(url: str, params: dict, timeout: int):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(weather, "WEATHER_API_KEY", "test-key")
    monkeypatch.setattr(weather.requests, "get", fake_get)

    result = weather.get_weather(latitude=41.3874, longitude=2.1686)

    assert result["error"] is None
    assert captured["params"]["q"] == "41.3874,2.1686"
    assert result["weather"]["ciudad"] == "Barcelona"


@pytest.mark.asyncio
async def test_realtime_weather_tool_uses_context_location_without_city(monkeypatch) -> None:
    import tool_registry

    captured: dict = {}

    def fake_get_weather(**kwargs):
        captured.update(kwargs)
        return {"error": None, "weather": {"ciudad": "Barcelona"}}

    monkeypatch.setattr(tool_registry, "get_weather", fake_get_weather)
    monkeypatch.setattr(
        tool_registry,
        "format_weather_for_chat",
        lambda data: f"Ciudad: {data['weather']['ciudad']}",
    )

    result = await tool_registry._tool_obtener_clima(
        {},
        {
            "user_profile": {"city": "Madrid"},
            "user_location": {"latitude": 41.3874, "longitude": 2.1686},
        },
    )

    assert captured == {
        "city": "",
        "country_code": "ES",
        "latitude": 41.3874,
        "longitude": 2.1686,
    }
    assert result == "Ciudad: Barcelona"
