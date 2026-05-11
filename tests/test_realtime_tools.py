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
    main.app.dependency_overrides[main.get_current_user_id] = lambda: "user-123"

    try:
        with TestClient(main.app) as client:
            response = client.post("/realtime/session", json={})
    finally:
        main.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["tool_call_handler"] == "backend"
    assert response.json()["voice"] == "ara"


def test_realtime_session_config_declares_audio_and_sensitive_vad() -> None:
    import main

    config = main._build_realtime_session({}, {}, {})

    assert config["audio"] == {
        "input": {"format": {"type": "audio/pcm", "rate": 24000}},
        "output": {"format": {"type": "audio/pcm", "rate": 24000}},
    }
    assert config["turn_detection"]["type"] == "server_vad"
    assert config["turn_detection"]["threshold"] == 0.35
