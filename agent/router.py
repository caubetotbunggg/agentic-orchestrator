"""
AgentRouter — uses Groq API (gpt-oss-20b / gemma2-9b-it) to classify user intent
and dispatch to the appropriate predefined graph.
"""
import json
import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

from agent.graphs import AVAILABLE_GRAPHS

load_dotenv()
logger = logging.getLogger("uvicorn")

# ---------------------------------------------------------------------------
# System prompt for intent classification
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a document-processing intent classifier.
Given a user request, output ONLY valid JSON (no explanation, no markdown) with:
{
  "graph": "<graph_name>",
  "src_lang": "<NLLB lang code>",
  "tgt_lang": "<NLLB lang code>"
}

Available graphs:
- "full_translate_graph": User wants to translate the full content/text/tables in the image.
- "ocr_only_graph": User wants to read/extract text from the image without translation.
- "table_extract_graph": User wants to see table data as a structured table (not translated).
- "translate_region_graph": User wants to translate a specific region/table/section.

Default languages: src_lang="eng_Latn", tgt_lang="vie_Latn" unless user specifies otherwise.
Common language codes: Vietnamese=vie_Latn, English=eng_Latn, French=fra_Latn,
  Chinese=zho_Hans, Japanese=jpn_Jpan, Korean=kor_Hang.
If unsure, use "ocr_only_graph"."""


# ---------------------------------------------------------------------------
# GroqRouter: singleton wrapper around Groq API
# ---------------------------------------------------------------------------

class GroqRouter:
    """Singleton wrapper around Groq API for intent routing."""
    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            try:
                api_key = os.environ.get("GROQ_API_KEY")
                if not api_key:
                    logger.warning("[Router] GROQ_API_KEY not found in environment. Routing may fail.")
                    
                logger.info("[Router] Initializing Groq API Client for Orchestrator...")
                cls._client = Groq(api_key=api_key)
                
            except Exception as e:
                logger.error(f"[Router] Failed to initialize Groq API Client: {e}")
                cls._client = None
        return cls._client


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

def _parse_router_output(raw: str) -> dict:
    """Extract JSON from the model output."""
    match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def classify_intent(
    prompt: str,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "vie_Latn",
) -> dict:
    """
    Classify the user prompt → returns dict with:
        graph: str, src_lang: str, tgt_lang: str
    Falls back to "ocr_only_graph" if model unavailable or parse fails.
    """
    fallback = {
        "graph": "ocr_only_graph",
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
    }

    client = GroqRouter.get_client()
    if client is None:
        logger.warning("[Router] Groq client unavailable, using fallback graph.")
        return fallback

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"User request: \"{prompt}\"\n"
                f"Default src_lang: {src_lang}, tgt_lang: {tgt_lang}\n"
                "Output JSON only:"
            ),
        },
    ]

    try:
        model = os.environ.get("GROQ_ORCHESTRATOR_MODEL", "gpt-oss-20b")
        outputs = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.0,
            max_completion_tokens=128,
        )
        
        assistant_msg = outputs.choices[0].message.content
        logger.info(f"[Router] Raw output: {assistant_msg[:200]}")
        parsed = _parse_router_output(assistant_msg)

        graph = parsed.get("graph", "")
        if graph not in AVAILABLE_GRAPHS:
            logger.warning(f"[Router] Unknown graph '{graph}', using fallback.")
            return fallback

        return {
            "graph": graph,
            "src_lang": parsed.get("src_lang", src_lang),
            "tgt_lang": parsed.get("tgt_lang", tgt_lang),
        }

    except Exception as e:
        logger.error(f"[Router] Classification error: {e}")
        return fallback
