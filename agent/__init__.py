"""
Agent entry point.
"""
import logging
from agent.router import classify_intent
from agent.graphs import get_graph
from agent.state import AgentState

logger = logging.getLogger("uvicorn")


def run_agent(
    image_bytes: bytes,
    prompt: str,
    src_lang: str = "eng_Latn",
    tgt_lang: str = "vie_Latn",
) -> dict:
    """
    Run the agent pipeline:
      1. Classify intent via Phi-3-mini router
      2. Dispatch to the appropriate predefined graph
      3. Return final_output + metadata

    Returns:
        {
            "output": str,
            "graph_used": str,
            "steps": list[str],
            "error": str | None,
        }
    """
    route = classify_intent(prompt, src_lang=src_lang, tgt_lang=tgt_lang)
    graph_name = route["graph"]
    resolved_src = route["src_lang"]
    resolved_tgt = route["tgt_lang"]

    logger.info(f"[Agent] intent={graph_name} | {resolved_src}→{resolved_tgt}")

    initial_state: AgentState = {
        "image_bytes": image_bytes,
        "prompt": prompt,
        "src_lang": resolved_src,
        "tgt_lang": resolved_tgt,
        "steps": [f"router → {graph_name}"],
    }

    try:
        graph = get_graph(graph_name)
        result: AgentState = graph.invoke(initial_state)

        return {
            "output": result.get("final_output", ""),
            "graph_used": graph_name,
            "steps": result.get("steps", []),
            "error": result.get("error"),
        }
    except Exception as e:
        logger.error(f"[Agent] Graph execution failed: {e}")
        return {
            "output": "",
            "graph_used": graph_name,
            "steps": initial_state["steps"],
            "error": str(e),
        }
