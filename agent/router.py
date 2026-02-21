"""
AgentRouter — uses local Phi-3-mini-4k-instruct to classify user intent
and dispatch to the appropriate predefined graph.
"""
import json
import logging
import re
from typing import Optional

import torch

from agent.graphs import AVAILABLE_GRAPHS

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
# PhiRouter: singleton wrapper around Phi-3-mini
# ---------------------------------------------------------------------------

class PhiRouter:
    """Singleton wrapper around Phi-3-mini pipeline for intent routing."""
    _pipeline = None

    @classmethod
    def get_pipeline(cls):
        if cls._pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline, AutoModelForCausalLM, AutoTokenizer

                model_name = "microsoft/Phi-3-mini-4k-instruct"
                logger.info(f"[Router] Loading Phi-3-mini from '{model_name}'...")

                device = "cpu"
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = 0

                dtype = torch.float16 if device != "cpu" else torch.float32

                # Load with attn_implementation="eager" to avoid DynamicCache/flash-attn
                # compatibility issues on MPS with transformers 4.x
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    attn_implementation="eager",
                )
                model = model.to(device)

                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                )

                cls._pipeline = hf_pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                )
                logger.info("[Router] Phi-3-mini loaded successfully.")
            except Exception as e:
                logger.error(f"[Router] Failed to load Phi-3-mini: {e}")
                cls._pipeline = None
        return cls._pipeline


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

    pipe = PhiRouter.get_pipeline()
    if pipe is None:
        logger.warning("[Router] Phi-3-mini unavailable, using fallback graph.")
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
        outputs = pipe(
            messages,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=False,     # Fix: disable DynamicCache to avoid seen_tokens error on MPS
        )
        # Phi-3 pipeline output: list of {generated_text: [...messages...]}
        generated = outputs[0]["generated_text"]
        if isinstance(generated, list):
            assistant_msg = next(
                (m["content"] for m in reversed(generated) if m.get("role") == "assistant"),
                "",
            )
        else:
            assistant_msg = str(generated)

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
