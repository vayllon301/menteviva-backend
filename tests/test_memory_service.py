from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memory_service import _expire_soft_facts


def _make_fact(text: str, category: str, created_at: str) -> dict:
    return {"text": text, "category": category, "created_at": created_at}


def _days_ago(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


class TestExpireSoftFacts:
    def test_expire_keeps_hard_facts_regardless_of_age(self) -> None:
        facts = [
            _make_fact("Se llama María", "hard", _days_ago(365)),
            _make_fact("Tiene diabetes", "hard", _days_ago(100)),
        ]
        result = _expire_soft_facts(facts)
        assert len(result) == 2
        assert all(f["category"] == "hard" for f in result)

    def test_expire_removes_old_soft_facts(self) -> None:
        facts = [
            _make_fact("Le gusta el fútbol", "soft", _days_ago(31)),
            _make_fact("Prefiere té", "soft", _days_ago(60)),
        ]
        result = _expire_soft_facts(facts)
        assert len(result) == 0

    def test_expire_keeps_soft_facts_within_30_days(self) -> None:
        facts = [
            _make_fact("Está contenta hoy", "soft", _days_ago(0)),
            _make_fact("Quiere ver las noticias", "soft", _days_ago(15)),
            _make_fact("Le duele la rodilla", "soft", _days_ago(29)),
        ]
        result = _expire_soft_facts(facts)
        assert len(result) == 3

    def test_expire_keeps_unparseable_facts(self) -> None:
        facts = [
            _make_fact("Dato misterioso", "soft", "not-a-date"),
            _make_fact("Otro dato", "soft", ""),
            {"text": "Sin created_at", "category": "soft"},
        ]
        result = _expire_soft_facts(facts)
        assert len(result) == 3
