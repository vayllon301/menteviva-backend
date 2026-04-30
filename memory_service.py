from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import httpx
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ExtractedFact(BaseModel):
    text: str
    category: str = Field(
        description="'hard' para hechos permanentes (nombre, familia, enfermedades) "
                    "o 'soft' para preferencias o estados temporales"
    )


class ExtractionResult(BaseModel):
    new_facts: List[ExtractedFact] = Field(default_factory=list)
    narrative_update: str = Field(
        default="",
        description="Breve resumen narrativo de lo nuevo aprendido sobre el usuario",
    )


class MergedFact(BaseModel):
    text: str
    category: str
    created_at: str


class MergeResult(BaseModel):
    facts: List[MergedFact] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM singleton
# ---------------------------------------------------------------------------

_llm_instance: Optional[ChatOpenAI] = None


def _get_llm() -> ChatOpenAI:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
    return _llm_instance


# ---------------------------------------------------------------------------
# Supabase helpers (httpx, no supabase-py)
# ---------------------------------------------------------------------------

def _supabase_headers() -> dict:
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _supabase_url() -> str:
    return os.getenv("SUPABASE_URL", "")


async def _load_memory(user_id: str) -> dict:
    """Load existing memory row for a user from Supabase."""
    url = f"{_supabase_url()}/rest/v1/user_memory?id=eq.{user_id}&select=*"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=_supabase_headers())
        resp.raise_for_status()
        data = resp.json()
        return data[0] if data else {}


async def _upsert_memory(memory: dict) -> None:
    """Upsert a memory row into Supabase."""
    url = f"{_supabase_url()}/rest/v1/user_memory"
    headers = {**_supabase_headers(), "Prefer": "resolution=merge-duplicates"}
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, headers=headers, json=memory)
        resp.raise_for_status()


# ---------------------------------------------------------------------------
# Expiry
# ---------------------------------------------------------------------------

def _expire_soft_facts(facts: List[dict]) -> List[dict]:
    """Remove soft facts older than 30 days. Keep hard facts always.

    Facts whose created_at cannot be parsed are kept as a fail-safe.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    kept: List[dict] = []
    for fact in facts:
        category = fact.get("category", "")
        if category == "hard":
            kept.append(fact)
            continue

        created_at_str = fact.get("created_at", "")
        try:
            created_at = datetime.fromisoformat(created_at_str)
            # Ensure timezone-aware comparison
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            if created_at >= cutoff:
                kept.append(fact)
            # else: soft fact is expired, drop it
        except (ValueError, TypeError):
            # Unparseable created_at — keep as fail-safe
            kept.append(fact)

    return kept


# ---------------------------------------------------------------------------
# Stage 1: Extract
# ---------------------------------------------------------------------------

async def _extract(messages: List[dict]) -> Optional[ExtractionResult]:
    """Use LLM to extract personal facts from recent conversation messages."""
    llm = _get_llm()
    structured_llm = llm.with_structured_output(ExtractionResult)

    conversation_text = "\n".join(
        f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages
    )

    prompt = (
        "Eres un asistente que extrae información personal del usuario a partir de "
        "una conversación. Analiza los mensajes y extrae hechos relevantes sobre el "
        "usuario.\n\n"
        "Clasifica cada hecho como:\n"
        "- 'hard': hechos permanentes como nombre, familia, enfermedades crónicas, "
        "dirección, fecha de nacimiento.\n"
        "- 'soft': preferencias temporales, estados de ánimo, intereses actuales.\n\n"
        "También genera un breve resumen narrativo de lo nuevo aprendido.\n\n"
        "Si no hay información personal nueva, devuelve listas vacías.\n\n"
        f"Conversación:\n{conversation_text}"
    )

    try:
        result = await structured_llm.ainvoke(prompt)
        return result
    except Exception:
        logger.exception("Stage 1 (extract) failed")
        return None


# ---------------------------------------------------------------------------
# Stage 2a: Merge
# ---------------------------------------------------------------------------

async def _merge(
    existing_facts: List[dict], new_facts: List[dict]
) -> Optional[MergeResult]:
    """Merge new facts into existing facts, resolving contradictions."""
    llm = _get_llm()
    structured_llm = llm.with_structured_output(MergeResult)

    prompt = (
        "Eres un asistente que fusiona hechos sobre un usuario. "
        "Tienes los hechos existentes y los hechos nuevos. "
        "Debes:\n"
        "1. Mantener todos los hechos que no se contradicen.\n"
        "2. Si un hecho nuevo contradice uno existente, quedarte con el nuevo.\n"
        "3. Eliminar duplicados.\n"
        "4. Conservar el campo 'created_at' original cuando el hecho se mantiene, "
        "y usar el 'created_at' del hecho nuevo cuando se reemplaza.\n\n"
        f"Hechos existentes:\n{existing_facts}\n\n"
        f"Hechos nuevos:\n{new_facts}\n\n"
        "Devuelve la lista fusionada completa."
    )

    try:
        result = await structured_llm.ainvoke(prompt)
        return result
    except Exception:
        logger.exception("Stage 2a (merge) failed")
        return None


# ---------------------------------------------------------------------------
# Stage 2b: Narrative rebuild
# ---------------------------------------------------------------------------

async def _rebuild_narrative(
    merged_facts: List[dict], narrative_update: str
) -> str:
    """Rebuild the user narrative from merged facts and the latest update."""
    llm = _get_llm()

    prompt = (
        "Eres un asistente que crea un resumen narrativo breve sobre un usuario "
        "mayor. A partir de los hechos conocidos y la actualización reciente, "
        "escribe un párrafo corto en tercera persona que describa al usuario.\n\n"
        f"Hechos conocidos:\n{merged_facts}\n\n"
        f"Actualización reciente:\n{narrative_update}\n\n"
        "Escribe el resumen narrativo:"
    )

    try:
        response = await llm.ainvoke(prompt)
        return response.content
    except Exception:
        logger.exception("Stage 2b (narrative) failed — using fallback")
        return narrative_update


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

async def run_memory_pipeline(user_id: str, messages: List[dict]) -> None:
    """Full two-stage memory pipeline, designed to run as a background task.

    Stage 1: Extract facts from the conversation.
    Stage 2a: Merge with existing facts.
    Stage 2b: Rebuild narrative.
    Finally: Upsert to Supabase.
    """
    # Stage 1 — Extract
    extraction = await _extract(messages)
    if extraction is None:
        logger.warning("Memory pipeline aborted: extraction failed")
        return

    if not extraction.new_facts and not extraction.narrative_update:
        logger.info("Memory pipeline: nothing new to store")
        return

    # Stamp created_at on new facts (Python-set, NOT LLM)
    now_iso = datetime.now(timezone.utc).isoformat()
    new_facts = [
        {
            "text": f.text,
            "category": f.category,
            "created_at": now_iso,
        }
        for f in extraction.new_facts
    ]

    # Load existing memory from Supabase
    try:
        existing_memory = await _load_memory(user_id)
    except Exception:
        logger.exception("Memory pipeline: failed to load existing memory")
        existing_memory = {}

    existing_facts: List[dict] = existing_memory.get("facts", [])

    # Expire soft facts
    existing_facts = _expire_soft_facts(existing_facts)

    # Stage 2a — Merge
    merge_result = await _merge(existing_facts, new_facts)
    if merge_result is None:
        logger.warning("Memory pipeline aborted: merge failed")
        return  # Do NOT continue to upsert

    merged_facts = [f.dict() for f in merge_result.facts]

    # Stage 2b — Narrative
    narrative = await _rebuild_narrative(merged_facts, extraction.narrative_update)

    # Upsert to Supabase
    memory_row = {
        "id": user_id,
        "facts": merged_facts,
        "narrative": narrative,
        "updated_at": now_iso,
    }

    try:
        await _upsert_memory(memory_row)
        logger.info("Memory pipeline: upserted memory for user %s", user_id)
    except Exception:
        logger.exception("Memory pipeline: upsert failed (silently discarded)")
