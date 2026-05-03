from fastapi.testclient import TestClient


def test_extract_realtime_function_call_from_output_item() -> None:
    import main

    payload = {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "name": "obtener_clima",
            "call_id": "call_123",
            "arguments": '{"ciudad":"Madrid"}',
        },
    }

    assert main._extract_realtime_function_call(payload) == {
        "name": "obtener_clima",
        "arguments": '{"ciudad":"Madrid"}',
        "call_id": "call_123",
    }


def test_extract_realtime_function_call_from_arguments_done() -> None:
    import main

    payload = {
        "type": "response.function_call_arguments.done",
        "name": "listar_recordatorios",
        "call_id": "call_456",
        "arguments": "{}",
    }

    assert main._extract_realtime_function_call(payload) == {
        "name": "listar_recordatorios",
        "arguments": "{}",
        "call_id": "call_456",
    }


def test_extract_realtime_function_call_returns_none_for_non_tool_event() -> None:
    import main

    payload = {"type": "response.text.done", "text": "hola"}
    assert main._extract_realtime_function_call(payload) is None


def test_realtime_session_defaults_to_backend_tool_handler(monkeypatch) -> None:
    import main

    monkeypatch.setattr(main, "_get_xai_api_key", lambda: "test-key")

    with TestClient(main.app) as client:
        response = client.post("/realtime/session", json={})

    assert response.status_code == 200
    assert response.json()["tool_call_handler"] == "backend"
