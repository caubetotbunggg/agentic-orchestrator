from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from typing import List
import io
from .schemas import LayoutResponse, OCRResponse, TranslateResponse, LayoutRegion, OCRTextItem, AgentResponse
from engine.services import detect_layout_service, ocr_service, crop_service, translate_service
from agent import run_agent

router = APIRouter()


@router.post("/layout-detect", response_model=LayoutResponse, summary="Detect layout regions in an image")
async def detect_layout(file: UploadFile = File(...)):
    """
    Detects layout regions using PP-DocLayoutV2 (via PP-Structure).
    """
    content = await file.read()
    try:
        regions = detect_layout_service(content)
        return LayoutResponse(regions=regions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ocr", response_model=OCRResponse, summary="Extract text from an image")
async def perform_ocr(file: UploadFile = File(...)):
    """
    Extracts text from the uploaded image using PP-OCRv5 (standardized to PP-OCRv4/latest).
    """
    content = await file.read()
    try:
        texts = ocr_service(content)
        return OCRResponse(texts=texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/translate", response_model=TranslateResponse, summary="Translate text/file")
async def translate_content(
    file: UploadFile = File(...), 
    src_lang: str = Form("eng_Latn", description="Source language code (NLLB)"),
    tgt_lang: str = Form("vie_Latn", description="Target language code (NLLB)")
):
    """
    Translates content from the uploaded file (text file expected).
    Default: English (eng_Latn) -> Vietnamese (vie_Latn) using local NLLB model.
    """
    content = await file.read()
    try:
        text = translate_service(content, src_lang, tgt_lang)
        return TranslateResponse(translated_text=text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/crop", summary="Crop image based on bbox")
async def crop_image(file: UploadFile = File(...), bbox: str = Form(..., description="JSON string of [x1, y1, x2, y2]")):
    """
    Crops the uploaded image to the specified bounding box.
    Returns: Image bytes (image/png).
    """
    content = await file.read()
    try:
        cropped_bytes = crop_service(content, bbox)
        return StreamingResponse(io.BytesIO(cropped_bytes), media_type="image/png")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agent", response_model=AgentResponse, summary="Agentic document Q&A")
async def agent_endpoint(
    file: UploadFile = File(...),
    prompt: str = Form(..., description="User's natural language request"),
    src_lang: str = Form("eng_Latn", description="Source language (NLLB code)"),
    tgt_lang: str = Form("vie_Latn", description="Target language (NLLB code)"),
):
    """
    Agentic document processing endpoint.

    Sends the uploaded image and a natural-language prompt to the agent.
    The agent classifies the intent using Phi-3-mini and dispatches to one
    of the predefined LangGraph graphs:

    - **full_translate_graph** — translate all content in the image
    - **ocr_only_graph** — extract text without translation
    - **table_extract_graph** — extract table as structured markdown
    - **translate_region_graph** — translate specific region/table

    Returns the final output text + which graph was used + execution steps.
    """
    content = await file.read()
    try:
        result = run_agent(content, prompt, src_lang=src_lang, tgt_lang=tgt_lang)
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        return AgentResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
