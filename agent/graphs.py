"""
Predefined LangGraph graphs for document processing.

Graphs:
    1. full_translate_graph   — layout → filter → crop → OCR → merge → translate
    2. ocr_only_graph         — OCR on full image
    3. table_extract_graph    — layout → filter(table) → crop → OCR → parse markdown table
    4. translate_region_graph — layout → pick best region → crop → OCR → translate
"""
import logging
from typing import List

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from tools.document_tools import (
    tool_layout_detect,
    tool_filter_regions,
    tool_crop,
    tool_ocr,
    tool_merge_text,
    tool_translate,
    tool_parse_table,
)

logger = logging.getLogger("uvicorn")


# ============================================================
# Helper: log step
# ============================================================

def _log(state: AgentState, msg: str) -> dict:
    steps = list(state.get("steps", []))
    steps.append(msg)
    return {"steps": steps}


# ============================================================
# Shared node implementations
# ============================================================

def node_layout_detect(state: AgentState) -> dict:
    logger.info("[Graph] node_layout_detect")
    regions = tool_layout_detect(state["image_bytes"])
    steps = list(state.get("steps", []))
    steps.append(f"layout_detect → {len(regions)} regions found")
    return {"regions": regions, "steps": steps}


def node_ocr_full(state: AgentState) -> dict:
    logger.info("[Graph] node_ocr_full")
    items = tool_ocr(state["image_bytes"])
    steps = list(state.get("steps", []))
    steps.append(f"ocr_full → {len(items)} text items")
    return {"ocr_results": [items], "steps": steps}


def node_merge_and_output_ocr(state: AgentState) -> dict:
    """Merge OCR results and set as final_output (for ocr_only_graph)."""
    all_items = []
    for items in state.get("ocr_results", []):
        all_items.extend(items)
    merged = tool_merge_text(all_items)
    steps = list(state.get("steps", []))
    steps.append("merge_text → done")
    return {"merged_text": merged, "final_output": merged, "steps": steps}


# ============================================================
# Graph 1: full_translate_graph
# ============================================================

def _ft_filter(state: AgentState) -> dict:
    regions = tool_filter_regions(state.get("regions", []), types=["text", "table", "title"])
    # Fallback: if nothing found, use all regions
    if not regions:
        regions = state.get("regions", [])
    steps = list(state.get("steps", []))
    steps.append(f"filter_regions(text,table,title) → {len(regions)} regions")
    return {"filtered_regions": regions, "steps": steps}


def _ft_crop(state: AgentState) -> dict:
    image = state["image_bytes"]
    crops = []
    for region in state.get("filtered_regions", []):
        try:
            crop = tool_crop(image, list(region.bbox))
            crops.append(crop)
        except Exception as e:
            logger.warning(f"[Graph] crop failed for region {region.bbox}: {e}")
    steps = list(state.get("steps", []))
    steps.append(f"crop → {len(crops)} crops")
    return {"cropped_images": crops, "steps": steps}


def _ft_ocr_crops(state: AgentState) -> dict:
    results = []
    for crop_bytes in state.get("cropped_images", []):
        try:
            items = tool_ocr(crop_bytes)
            results.append(items)
        except Exception as e:
            logger.warning(f"[Graph] ocr crop failed: {e}")
            results.append([])
    steps = list(state.get("steps", []))
    steps.append(f"ocr_crops → {sum(len(r) for r in results)} text items total")
    return {"ocr_results": results, "steps": steps}


def _ft_merge(state: AgentState) -> dict:
    all_items = []
    for items in state.get("ocr_results", []):
        all_items.extend(items)
    merged = tool_merge_text(all_items)
    steps = list(state.get("steps", []))
    steps.append("merge_text → done")
    return {"merged_text": merged, "steps": steps}


def _ft_translate(state: AgentState) -> dict:
    text = state.get("merged_text", "")
    src = state.get("src_lang", "eng_Latn")
    tgt = state.get("tgt_lang", "vie_Latn")
    translated = tool_translate(text, src_lang=src, tgt_lang=tgt)
    steps = list(state.get("steps", []))
    steps.append(f"translate({src}→{tgt}) → done")
    return {"translated_text": translated, "final_output": translated, "steps": steps}


def build_full_translate_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("layout_detect", node_layout_detect)
    g.add_node("filter_regions", _ft_filter)
    g.add_node("crop", _ft_crop)
    g.add_node("ocr_crops", _ft_ocr_crops)
    g.add_node("merge_text", _ft_merge)
    g.add_node("translate", _ft_translate)

    g.set_entry_point("layout_detect")
    g.add_edge("layout_detect", "filter_regions")
    g.add_edge("filter_regions", "crop")
    g.add_edge("crop", "ocr_crops")
    g.add_edge("ocr_crops", "merge_text")
    g.add_edge("merge_text", "translate")
    g.add_edge("translate", END)
    return g.compile()


# ============================================================
# Graph 2: ocr_only_graph
# ============================================================

def build_ocr_only_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("ocr_full", node_ocr_full)
    g.add_node("merge_output", node_merge_and_output_ocr)

    g.set_entry_point("ocr_full")
    g.add_edge("ocr_full", "merge_output")
    g.add_edge("merge_output", END)
    return g.compile()


# ============================================================
# Graph 3: table_extract_graph
# ============================================================

def _te_filter_table(state: AgentState) -> dict:
    regions = tool_filter_regions(state.get("regions", []), types=["table"])
    if not regions:
        regions = state.get("regions", [])
    steps = list(state.get("steps", []))
    steps.append(f"filter_regions(table) → {len(regions)} regions")
    return {"filtered_regions": regions, "steps": steps}


def _te_crop_table(state: AgentState) -> dict:
    image = state["image_bytes"]
    crops = []
    for region in state.get("filtered_regions", []):
        try:
            crop = tool_crop(image, list(region.bbox))
            crops.append(crop)
        except Exception as e:
            logger.warning(f"[Graph] crop table failed: {e}")
    steps = list(state.get("steps", []))
    steps.append(f"crop_table → {len(crops)} crops")
    return {"cropped_images": crops, "steps": steps}


def _te_ocr_table(state: AgentState) -> dict:
    results = []
    for crop_bytes in state.get("cropped_images", []):
        try:
            items = tool_ocr(crop_bytes)
            results.append(items)
        except Exception as e:
            logger.warning(f"[Graph] ocr table crop failed: {e}")
            results.append([])
    steps = list(state.get("steps", []))
    steps.append(f"ocr_table → {sum(len(r) for r in results)} text items")
    return {"ocr_results": results, "steps": steps}


def _te_parse_table(state: AgentState) -> dict:
    all_items = []
    for items in state.get("ocr_results", []):
        all_items.extend(items)
    table_md = tool_parse_table(all_items)
    steps = list(state.get("steps", []))
    steps.append("parse_table → markdown table")
    return {"final_output": table_md, "steps": steps}


def build_table_extract_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("layout_detect", node_layout_detect)
    g.add_node("filter_table", _te_filter_table)
    g.add_node("crop_table", _te_crop_table)
    g.add_node("ocr_table", _te_ocr_table)
    g.add_node("parse_table", _te_parse_table)

    g.set_entry_point("layout_detect")
    g.add_edge("layout_detect", "filter_table")
    g.add_edge("filter_table", "crop_table")
    g.add_edge("crop_table", "ocr_table")
    g.add_edge("ocr_table", "parse_table")
    g.add_edge("parse_table", END)
    return g.compile()


# ============================================================
# Graph 4: translate_region_graph
# ============================================================

def _tr_filter_text_table(state: AgentState) -> dict:
    regions = tool_filter_regions(state.get("regions", []), types=["text", "table", "title"])
    if not regions:
        regions = state.get("regions", [])
    # Pick the largest region by area as "best candidate"
    if regions:
        regions = sorted(
            regions,
            key=lambda r: (r.bbox[2] - r.bbox[0]) * (r.bbox[3] - r.bbox[1]),
            reverse=True,
        )
    steps = list(state.get("steps", []))
    steps.append(f"filter_regions(text,table,title) → picked largest {len(regions)} regions")
    return {"filtered_regions": regions, "steps": steps}


def _tr_crop_best(state: AgentState) -> dict:
    """Crop all filtered regions (sorted largest first)."""
    image = state["image_bytes"]
    crops = []
    for region in state.get("filtered_regions", []):
        try:
            crop = tool_crop(image, list(region.bbox))
            crops.append(crop)
        except Exception as e:
            logger.warning(f"[Graph] crop region failed: {e}")
    steps = list(state.get("steps", []))
    steps.append(f"crop_best → {len(crops)} crops")
    return {"cropped_images": crops, "steps": steps}


def _tr_ocr_region(state: AgentState) -> dict:
    results = []
    for crop_bytes in state.get("cropped_images", []):
        try:
            items = tool_ocr(crop_bytes)
            results.append(items)
        except Exception as e:
            logger.warning(f"[Graph] ocr region failed: {e}")
            results.append([])
    steps = list(state.get("steps", []))
    steps.append(f"ocr_region → {sum(len(r) for r in results)} text items")
    return {"ocr_results": results, "steps": steps}


def build_translate_region_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("layout_detect", node_layout_detect)
    g.add_node("filter_regions", _tr_filter_text_table)
    g.add_node("crop_best", _tr_crop_best)
    g.add_node("ocr_region", _tr_ocr_region)
    g.add_node("merge_text", _ft_merge)      # reuse from graph 1
    g.add_node("translate", _ft_translate)    # reuse from graph 1

    g.set_entry_point("layout_detect")
    g.add_edge("layout_detect", "filter_regions")
    g.add_edge("filter_regions", "crop_best")
    g.add_edge("crop_best", "ocr_region")
    g.add_edge("ocr_region", "merge_text")
    g.add_edge("merge_text", "translate")
    g.add_edge("translate", END)
    return g.compile()


# ============================================================
# Registry: compiled graph instances (lazy initialized)
# ============================================================

_GRAPHS: dict = {}


def get_graph(name: str):
    """Return a compiled graph by name. Builds it if not yet cached."""
    global _GRAPHS
    if name not in _GRAPHS:
        builders = {
            "full_translate_graph": build_full_translate_graph,
            "ocr_only_graph": build_ocr_only_graph,
            "table_extract_graph": build_table_extract_graph,
            "translate_region_graph": build_translate_region_graph,
        }
        if name not in builders:
            raise ValueError(f"Unknown graph: {name}. Available: {list(builders.keys())}")
        _GRAPHS[name] = builders[name]()
    return _GRAPHS[name]


AVAILABLE_GRAPHS = [
    "full_translate_graph",
    "ocr_only_graph",
    "table_extract_graph",
    "translate_region_graph",
]
