"""
AgentState — shared state passed between nodes in a LangGraph graph.
"""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from api.schemas import LayoutRegion, OCRTextItem


class AgentState(TypedDict, total=False):
    # ---- Input ----
    image_bytes: bytes                  # Raw bytes of the uploaded image
    prompt: str                         # User's natural-language request
    src_lang: str                       # Source language (NLLB code), default eng_Latn
    tgt_lang: str                       # Target language (NLLB code), default vie_Latn

    # ---- Router output ----
    intent: str                         # Chosen graph name, e.g. "full_translate_graph"

    # ---- Intermediate results ----
    regions: List[LayoutRegion]         # All detected layout regions
    filtered_regions: List[LayoutRegion]  # Regions after type filtering
    cropped_images: List[bytes]         # Cropped image bytes (one per region)
    ocr_results: List[List[OCRTextItem]]  # OCR results per cropped image

    # ---- Aggregated ----
    merged_text: str                    # All OCR text merged into one string
    translated_text: str                # Final translated text

    # ---- Final output ----
    final_output: str                   # The answer returned to the user
    steps: List[str]                    # Log of steps executed (for debugging)
    error: Optional[str]                # Error message if something failed

    # ---- Planner Agent State ----
    messages: Annotated[list[BaseMessage], add_messages] # Message history for LLM planner
    is_last_step: bool                  # LangGraph prebuilt require this
    remaining_steps: int                # LangGraph prebuilt require this
