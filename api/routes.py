from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from .schemas import LayoutResponse, OCRResponse, TranslateResponse, LayoutRegion, OCRTextItem

router = APIRouter()

@router.post("/layout-detect", response_model=LayoutResponse, summary="Detect layout regions in an image")
async def detect_layout(file: UploadFile = File(...)):
    """
    Detects layout regions (tables, figures, etc.) in the uploaded image.
    """
    # TODO: Implement actual layout detection logic
    # Mock response for now
    return LayoutResponse(
        regions=[
            LayoutRegion(type="table", bbox=(10, 10, 200, 200), confidence=0.95),
            LayoutRegion(type="text", bbox=(10, 210, 200, 300), confidence=0.98),
        ]
    )

@router.post("/ocr", response_model=OCRResponse, summary="Extract text from an image")
async def perform_ocr(file: UploadFile = File(...)):
    """
    Extracts text from the uploaded image using OCR.
    """
    # TODO: Implement actual OCR logic
    # Mock response for now
    return OCRResponse(
        texts=[
            OCRTextItem(text="Sample Text", bbox=(50, 50, 100, 80), confidence=0.99)
        ]
    )

@router.post("/translate", response_model=TranslateResponse, summary="Translate text/file")
async def translate_content(file: UploadFile = File(...)):
    """
    Translates content from the uploaded file.
    Note: Currently assumes file input as requested. 
    """
    # TODO: Implement actual translation logic
    # Mock response for now
    return TranslateResponse(translated_text="Xin chào (translated content)")
