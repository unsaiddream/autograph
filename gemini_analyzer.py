import os
import json
import base64
import re
import google.generativeai as genai
from PIL import Image
import io

GEMINI_PROMPT_TEMPLATE = """
Ты — система анализа юридических документов.
Проанализируй изображение страницы документа и найди все места где требуется:

1. ПОДПИСЬ — линии вида "________", слова "Подпись:", "Signature:", пустая строка после должности/ФИО
2. ПЕЧАТЬ — пометки "М.П.", "Место печати", "Stamp", пустой круг/квадрат, "L.S."
3. ФИО — строки после "ФИО:", "Ф.И.О.:", "Фамилия И.О.:", пустая строка рядом с подписью
4. ДАТА — строки "Дата:", "Date:", шаблон "__.__.__", "«__» _______ 20__ г."

Изображение имеет размер {width}x{height} пикселей.

Верни ТОЛЬКО валидный JSON без markdown-обёртки, без пояснений:
{{
  "page": 1,
  "zones": [
    {{
      "id": "sign_1",
      "type": "signature",
      "x": 150, "y": 520, "width": 220, "height": 70,
      "label": "подпись руководителя",
      "confidence": 0.95
    }},
    {{
      "id": "stamp_1",
      "type": "stamp",
      "x": 80, "y": 490, "width": 130, "height": 130,
      "label": "место печати",
      "confidence": 0.90
    }},
    {{
      "id": "name_1",
      "type": "fullname",
      "x": 150, "y": 500, "width": 280, "height": 35,
      "label": "ФИО руководителя",
      "confidence": 0.88
    }},
    {{
      "id": "date_1",
      "type": "date",
      "x": 380, "y": 500, "width": 160, "height": 35,
      "label": "дата подписания",
      "confidence": 0.92
    }}
  ]
}}

Если зон не найдено — верни {{"page": 1, "zones": []}}.
Координаты отсчитываются от левого верхнего угла.
"""


def _configure_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-pro")


def _extract_json(text: str) -> dict:
    text = text.strip()
    # Remove markdown code blocks if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    return json.loads(text)


def analyze_page(page_b64: str, page_number: int, width: int, height: int) -> dict:
    """
    Send a page image to Gemini Vision and return detected zones.
    Returns dict with 'page' and 'zones' keys.
    """
    model = _configure_gemini()

    prompt = GEMINI_PROMPT_TEMPLATE.format(width=width, height=height)

    # Decode base64 image
    image_data = base64.b64decode(page_b64)
    image = Image.open(io.BytesIO(image_data))

    # Convert to RGB if needed
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    response = model.generate_content([prompt, image])
    raw_text = response.text

    try:
        result = _extract_json(raw_text)
        result["page"] = page_number
        # Ensure zones is a list
        if "zones" not in result:
            result["zones"] = []
        return result
    except (json.JSONDecodeError, KeyError):
        return {"page": page_number, "zones": []}


def analyze_all_pages(pages_b64: list[str], page_sizes: list[tuple[int, int]]) -> list[dict]:
    """
    Analyze all pages and return list of zone data per page.
    """
    results = []
    for i, (page_b64, (width, height)) in enumerate(zip(pages_b64, page_sizes)):
        try:
            result = analyze_page(page_b64, i + 1, width, height)
        except Exception as e:
            result = {"page": i + 1, "zones": [], "error": str(e)}
        results.append(result)
    return results
