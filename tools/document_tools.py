"""
Document processing tool functions.
Wraps engine/services.py for use inside LangGraph graph nodes.
"""
import json
import logging
import time
from contextlib import contextmanager
from typing import List, Optional

from api.schemas import LayoutRegion, OCRTextItem
from engine.services import (
    detect_layout_service,
    ocr_service,
    crop_service,
    translate_service,
)

logger = logging.getLogger("uvicorn")


@contextmanager
def _timed(name: str):
    t0 = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"[Tool] {name} — {elapsed:.0f}ms")



# ---------------------------------------------------------------------------
# Tool: Layout Detection
# ---------------------------------------------------------------------------

def tool_layout_detect(image_bytes: bytes) -> List[LayoutRegion]:
    """Detect layout regions (text, table, figure, ...) in the image."""
    with _timed("layout_detect"):
        result = detect_layout_service(image_bytes)
    return result


# ---------------------------------------------------------------------------
# Tool: Filter Regions
# ---------------------------------------------------------------------------

def tool_filter_regions(
    regions: List[LayoutRegion],
    types: Optional[List[str]] = None,
) -> List[LayoutRegion]:
    """Filter layout regions by type (e.g. ['table', 'text']).
    If types is None or empty, return all regions.
    """
    with _timed(f"filter_regions(types={types})"):
        if not types:
            return regions
        types_lower = [t.lower() for t in types]
        result = [r for r in regions if r.type.lower() in types_lower]
    return result


# ---------------------------------------------------------------------------
# Tool: Crop
# ---------------------------------------------------------------------------

def tool_crop(image_bytes: bytes, bbox: List[int]) -> bytes:
    """Crop the image to the given [x1, y1, x2, y2] bounding box."""
    bbox_str = json.dumps(bbox)
    with _timed(f"crop(bbox={bbox})"):
        result = crop_service(image_bytes, bbox_str)
    return result


# ---------------------------------------------------------------------------
# Tool: OCR
# ---------------------------------------------------------------------------

def tool_ocr(image_bytes: bytes) -> List[OCRTextItem]:
    """Extract text items (text + bbox + confidence) from the image."""
    with _timed("ocr"):
        result = ocr_service(image_bytes)
    return result


# ---------------------------------------------------------------------------
# Tool: Merge text
# ---------------------------------------------------------------------------

def tool_merge_text(items: List[OCRTextItem]) -> str:
    """Merge OCR text items into a single string, ordered top-to-bottom."""
    with _timed("merge_text"):
        sorted_items = sorted(items, key=lambda x: (x.bbox[1], x.bbox[0]))
        result = " ".join(item.text for item in sorted_items if item.text.strip())
    return result


# ---------------------------------------------------------------------------
# Tool: Translate
# ---------------------------------------------------------------------------

def tool_translate(text: str, src_lang: str = "eng_Latn", tgt_lang: str = "vie_Latn") -> str:
    """Translate text using the local NLLB-200 model."""
    if not text.strip():
        return ""
    with _timed(f"translate({src_lang}→{tgt_lang})"):
        result = translate_service(text.encode("utf-8"), src_lang, tgt_lang)
    return result


# ---------------------------------------------------------------------------
# Tool: Parse table (simple markdown)
# ---------------------------------------------------------------------------

def tool_parse_table(items: List[OCRTextItem]) -> str:
    """
    Attempt to reconstruct a markdown table from OCR items.
    Uses row clustering by Y-coordinate proximity.
    """
    with _timed("parse_table"):
        if not items:
            return ""

        # Cluster into rows by Y proximity (threshold = avg height of items)
        avg_height = sum((i.bbox[3] - i.bbox[1]) for i in items) / max(len(items), 1)
        threshold = max(avg_height * 0.8, 10)

        rows: List[List[OCRTextItem]] = []
        for item in sorted(items, key=lambda x: (x.bbox[1], x.bbox[0])):
            placed = False
            for row in rows:
                row_y = sum((r.bbox[1] + r.bbox[3]) / 2 for r in row) / len(row)
                item_y = (item.bbox[1] + item.bbox[3]) / 2
                if abs(item_y - row_y) <= threshold:
                    row.append(item)
                    placed = True
                    break
            if not placed:
                rows.append([item])

        if not rows:
            return ""

        # Sort each row's items by X
        for row in rows:
            row.sort(key=lambda x: x.bbox[0])

        max_cols = max(len(row) for row in rows)

        def pad_row(row_items: List[OCRTextItem], n: int) -> List[str]:
            texts = [item.text for item in row_items]
            return texts + [""] * (n - len(texts))

        lines = []
        for idx, row in enumerate(rows):
            cells = pad_row(row, max_cols)
            lines.append("| " + " | ".join(cells) + " |")
            if idx == 0:
                lines.append("|" + "|".join(["---"] * max_cols) + "|")

        return "\n".join(lines)

