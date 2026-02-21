import io
import json
import logging
import numpy as np
from PIL import Image
try:
    from paddleocr import PaddleOCR, PPStructure
    _import_error = None
except ImportError as e:
    # Handle missing dependency gracefully during build/install phase
    PaddleOCR = None
    PPStructure = None
    _import_error = e

try:
    from groq import Groq
    import os
    from dotenv import load_dotenv
    load_dotenv()
    _groq_error = None
except ImportError as e:
    Groq = None
    _groq_error = e

from api.schemas import LayoutRegion, OCRTextItem

# Configure logging
logger = logging.getLogger("uvicorn")

class AIEngine:
    _layout_engine = None
    _ocr_engine = None
    _translate_client = None

    @classmethod
    def get_layout_engine(cls):
        if cls._layout_engine is None:
            if PPStructure is None:
                raise ImportError(f"paddleocr import failed: {_import_error}")
            logger.info("Initializing PP-Structure (Layout Engine)...")
            # Using 'PP-StructureV2' for layout analysis (covers PP-DocLayoutV2 logic)
            cls._layout_engine = PPStructure(
                show_log=False, 
                image_orientation=True,
                structure_version='PP-StructureV2'
            )
        return cls._layout_engine

    @classmethod
    def get_ocr_engine(cls):
        if cls._ocr_engine is None:
            if PaddleOCR is None:
                raise ImportError(f"paddleocr import failed: {_import_error}")
            logger.info("Initializing PaddleOCR (OCR Engine)...")
            # Using PP-OCRv4 as a stable choice. Validates if v5 is available implicitly or via update.
            cls._ocr_engine = PaddleOCR(
                use_angle_cls=True, 
                lang='en', 
                show_log=False,
                ocr_version='PP-OCRv4'
            )
        return cls._ocr_engine

    @classmethod
    def get_translate_engine(cls):
        if cls._translate_client is None:
            if Groq is None:
                raise ImportError(f"groq import failed: {_groq_error}")
            
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                logger.warning("GROQ_API_KEY not found in environment. Translation may fail.")
                
            logger.info("Initializing Groq API Client for Translation...")
            cls._translate_client = Groq(api_key=api_key)
            
        return cls._translate_client

def detect_layout_service(image_file: bytes) -> list[LayoutRegion]:
    try:
        engine = AIEngine.get_layout_engine()
        image = Image.open(io.BytesIO(image_file)).convert("RGB")
        img_array = np.array(image)
        
        # Run layout analysis
        result = engine(img_array)
        
        regions = []
        # Result is a list of dicts.
        for element in result:
            bbox = element.get('bbox')
            region_type = element.get('type')
            score = element.get('score') # Some versions return score specific to type classification
            
            # If explicit score not found, default to high confidence if detected
            confidence = float(score) if score is not None else 0.95

            if bbox and region_type:
                # Ensure bbox is integer tuple
                x1, y1, x2, y2 = map(int, bbox)
                
                regions.append(LayoutRegion(
                    type=region_type,
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence
                ))
        return regions
    except Exception as e:
        logger.error(f"Layout detection failed: {e}")
        # In production, might want to raise HTTPException
        raise e

def ocr_service(image_file: bytes) -> list[OCRTextItem]:
    try:
        ocr = AIEngine.get_ocr_engine()
        image = Image.open(io.BytesIO(image_file)).convert("RGB")
        img_array = np.array(image)
        
        # Run OCR
        # result structure: [ [ [box, (text, score)], ... ] ]
        result = ocr.ocr(img_array, cls=True)
        
        items = []
        if result and result[0]:
            for line in result[0]:
                coords = line[0] # [[x1,y1], [x2,y1], [x2, y2], [x1, y2]]
                text, score = line[1]
                
                # Calculate bounding box [x1, y1, x2, y2]
                xs = [pt[0] for pt in coords]
                ys = [pt[1] for pt in coords]
                x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
                
                items.append(OCRTextItem(
                    text=text,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(score)
                ))
        return items
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise e

def crop_service(image_file: bytes, bbox_str: str) -> bytes:
    try:
        bbox = json.loads(bbox_str)
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Bbox must be a list of 4 integers [x1, y1, x2, y2]")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON string for bbox")

    try:
        image = Image.open(io.BytesIO(image_file))
        x1, y1, x2, y2 = map(int, bbox)
        
        # Check bounds
        width, height = image.size
        # Clamp coordinates
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        if x2 <= x1 or y2 <= y1:
            # If invalid crop area, maybe return original or raise error?
            # Raising error is better layout logic debugging
            raise ValueError(f"Invalid crop dimensions: {x1},{y1},{x2},{y2}")

        cropped = image.crop((x1, y1, x2, y2))
        
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Crop failed: {e}")
        raise e

def translate_service(image_file: bytes, src_lang: str = "eng_Latn", tgt_lang: str = "vie_Latn") -> str:
    try:
        client = AIEngine.get_translate_engine()
        
        # Read text content.
        try:
            text = image_file.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("Input file must be text (utf-8) for translation service.")

        if not text.strip():
            return ""

        # Construct prompt
        system_prompt = f"""You are a professional translation engine.

Task:
Translate the given text from {src_lang} to {tgt_lang}.

Rules:
- Preserve the original meaning exactly.
- Do NOT add explanations.
- Do NOT add commentary.
- Do NOT summarize.
- Do NOT change formatting unless necessary.
- Keep names, numbers, URLs, and code unchanged.
- If the input text is empty, return an empty string.

Return ONLY the translated text.

Text:
\"\"\"
{text}
\"\"\"
"""
        
        model = os.environ.get("GROQ_MODEL")
        
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": system_prompt,
                }
            ],
            model=model,
            temperature=0.0,
            max_completion_tokens=4096,
        )
        
        translated_text = response.choices[0].message.content
        return translated_text.strip()


    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise e
