from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel, Field

# --- Layout Detect Models ---
class LayoutRegion(BaseModel):
    type: str = Field(..., description="Type of the region (e.g., 'table', 'text', 'figure')")
    bbox: Tuple[int, int, int, int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Confidence score (0.0 - 1.0)")

class LayoutResponse(BaseModel):
    regions: List[LayoutRegion]

# --- OCR Models ---
class OCRTextItem(BaseModel):
    text: str
    bbox: Tuple[int, int, int, int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Confidence score (0.0 - 1.0)")

class OCRResponse(BaseModel):
    texts: List[OCRTextItem]

# --- Translate Models ---
class TranslateResponse(BaseModel):
    translated_text: str
