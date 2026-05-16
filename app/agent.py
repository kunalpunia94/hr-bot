"""Core agent orchestration: intent guardrails → retrieval → Groq LLM → validation."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from groq import Groq

from app.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    MAX_RECOMMENDATIONS,
    MAX_TURNS,
    TOP_K_CONTEXT,
    TOP_K_RETRIEVAL,
)
from app.prompts import (
    REFUSAL_INJECTION,
    REFUSAL_LEGAL,
    REFUSAL_OFF_TOPIC,
    SYSTEM_PROMPT,
)
from app.retriever import get_retriever

logger = logging.getLogger(__name__)


# Keyword sets distilled from C7 (legal HIPAA refusal) and common
# prompt-injection / off-topic vectors.
LEGAL_KEYWORDS = {
    "legally required",
    "legal requirement",
    "lawsuit",
    "discrimination claim",
    "hipaa requirement",
    "compliance obligation",
    "regulatory obligation",
    "labor law",
    "employment law",
    "wrongful termination",
}

OFF_TOPIC_KEYWORDS = {
    "salary",
    "compensation band",
    "how much should i pay",
    "negotiate offer",
    "visa sponsorship",
    "immigration",
    "fire an employee",
    "terminate an employee",
    "layoff",
    "age discrimination",
}

INJECTION_KEYWORDS = {
    "ignore previous instructions",
    "ignore your instructions",
    "ignore all previous",
    "disregard previous",
    "forget your instructions",
    "you are now",
    "pretend you are",
    "act as if you",
    "jailbreak",
    "system prompt",
    "reveal your prompt",
}

TEST_TYPE_CODES = {
    "ability & aptitude": "A",
    "biodata & situational judgment": "B",
    "biodata & situational judgement": "B",
    "competencies": "C",
    "development & 360": "D",
    "assessment exercises": "E",
    "knowledge & skills": "K",
    "personality & behavior": "P",
    "personality & behaviour": "P",
    "simulations": "S",
}


_groq_client: Groq | None = None


def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY or GROQ_API_KEY == "xyz":
            raise RuntimeError(
                "GROQ_API_KEY is not set. Edit .env and paste your key "
                "from https://console.groq.com/keys"
            )
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _contains_any(text: str, keywords: set[str]) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def detect_guardrail(messages: list[dict]) -> str | None:
    """Return a refusal category string if the latest user message trips a
    guardrail, else None. Detected BEFORE LLM call so we save tokens and
    guarantee a safe, consistent refusal."""
    text = _last_user_text(messages)
    if _contains_any(text, INJECTION_KEYWORDS):
        return "injection"
    if _contains_any(text, LEGAL_KEYWORDS):
        return "legal"
    if _contains_any(text, OFF_TOPIC_KEYWORDS):
        return "off_topic"
    return None


def build_search_query(messages: list[dict]) -> str:
    """Concatenate the last few user turns for retrieval — gives the embedding
    model a richer signal than just the latest message."""
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    return " \n".join(user_msgs[-3:])


def format_history(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)


def keys_to_test_type(keys: list[str]) -> str:
    if not keys:
        return "K"
    codes: list[str] = []
    seen: set[str] = set()
    for k in keys:
        code = TEST_TYPE_CODES.get(k.lower().strip())
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    return ",".join(codes) if codes else "K"


def format_catalog_context(items: list[dict], limit: int = TOP_K_CONTEXT) -> str:
    """Compact, token-efficient YAML-ish rendering. Each item ~120 tokens."""
    lines = []
    for item in items[:limit]:
        keys = item.get("keys") or []
        test_type = keys_to_test_type(keys)
        levels = ", ".join((item.get("job_levels") or [])[:4])
        description = (item.get("description") or "").strip().replace("\n", " ")
        if len(description) > 180:
            description = description[:177] + "..."
        lines.append(
            f"- name: {item.get('name', '')}\n"
            f"  url: {item.get('link', '')}\n"
            f"  test_type: {test_type}\n"
            f"  levels: {levels}\n"
            f"  duration: {item.get('duration') or '—'}\n"
            f"  desc: {description}"
        )
    return "\n".join(lines)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """Pull the first balanced JSON object out of an LLM response."""
    if not text:
        return None
    # Strip common code-fence wrappers
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```\s*$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(stripped)
    if not match:
        return None
    try:
        return json.loads(match.group(), strict=False)
    except json.JSONDecodeError:
        return None


def validate_recommendations(raw_recs: Any) -> list[dict]:
    """Drop fabricated URLs; reconcile by name when possible; cap at 10."""
    retriever = get_retriever()
    if not isinstance(raw_recs, list):
        return []
    cleaned: list[dict] = []
    seen_urls: set[str] = set()
    for rec in raw_recs:
        if not isinstance(rec, dict):
            continue
        name = (rec.get("name") or "").strip()
        url = (rec.get("url") or "").strip()
        test_type = (rec.get("test_type") or "").strip()

        item = None
        if url and retriever.is_valid_url(url):
            item = retriever.get_by_url(url)
        elif name:
            item = retriever.get_by_name(name)

        if item is None:
            logger.warning(
                "Dropping fabricated recommendation: name=%r url=%r", name, url
            )
            continue

        canonical_url = item["link"]
        if canonical_url in seen_urls:
            continue
        seen_urls.add(canonical_url)

        cleaned.append(
            {
                "name": item["name"],
                "url": canonical_url,
                "test_type": test_type or keys_to_test_type(item.get("keys") or []),
            }
        )
        if len(cleaned) >= MAX_RECOMMENDATIONS:
            break
    return cleaned


def _refusal_response(category: str) -> dict:
    reply = {
        "injection": REFUSAL_INJECTION,
        "legal": REFUSAL_LEGAL,
        "off_topic": REFUSAL_OFF_TOPIC,
    }.get(category, REFUSAL_OFF_TOPIC)
    return {
        "reply": reply,
        "recommendations": [],
        "end_of_conversation": False,
    }


def chat(messages: list[dict]) -> dict:
    """Single-turn agent step. Always returns a dict matching ChatResponse."""
    if not messages:
        return {
            "reply": "Tell me about the role you're hiring for and I'll suggest assessments.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    # Hard cap conversation length the model sees (the API also enforces this).
    trimmed = messages[-MAX_TURNS:]

    guardrail = detect_guardrail(trimmed)
    if guardrail:
        logger.info("Guardrail triggered: %s", guardrail)
        return _refusal_response(guardrail)

    retriever = get_retriever()
    query = build_search_query(trimmed)
    candidates = retriever.search(query, top_k=TOP_K_RETRIEVAL)

    prompt = SYSTEM_PROMPT.format(
        catalog_context=format_catalog_context(candidates),
        history=format_history(trimmed),
    )

    try:
        client = _get_client()
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500,
            response_format={"type": "json_object"},
        )
        raw = (completion.choices[0].message.content or "").strip()
    except Exception as exc:  # pragma: no cover - depends on live API
        logger.exception("Groq call failed: %s", exc)
        return {
            "reply": "I hit a temporary issue reaching the language model. Please try again in a moment.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    parsed = extract_json(raw)
    if parsed is None:
        logger.warning("Could not parse JSON from model output: %r", raw[:300])
        return {
            "reply": "Sorry, I didn't format that correctly. Could you rephrase your last request?",
            "recommendations": [],
            "end_of_conversation": False,
        }

    recs = validate_recommendations(parsed.get("recommendations"))
    return {
        "reply": str(parsed.get("reply") or "").strip()
        or "Here's what I'd suggest based on the catalog.",
        "recommendations": recs,
        "end_of_conversation": bool(parsed.get("end_of_conversation", False)),
    }
