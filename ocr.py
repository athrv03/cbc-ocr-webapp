from __future__ import annotations

import base64
import re
from io import BytesIO
from typing import Any

import fitz
import requests
from PIL import Image

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "glm-ocr"


class OllamaConnectionError(RuntimeError):
    pass


class OllamaModelNotFoundError(RuntimeError):
    pass


# Logic based on input_pdf_to_json.ipynb
STANDARD_MAP: dict[str, str] = {
    "W.B.C.Count": "WBC",
    "Neutrophils": "NE%",
    "Lymphocytes": "LY%",
    "Monocytes": "MO%",
    "Eosinophils": "EO%",
    "Basophils": "BA%",
    "Absolute Neutrophil Count": "NE#",
    "Absolute Lymphocyte Count": "LY#",
    "Absolute Monocyte Count": "MO#",
    "Absolute Eosinophil Count": "EO#",
    "Absolute Basophil Count": "BA#",
    "R.B.C Count": "RBC",
    "Haemoglobin": "HGB",
    "Haematocrit": "HCT",
    "MCV": "MCV",
    "MCH": "MCH",
    "MCHC": "MCHC",
    "RDW": "RDW",
    "Platelet Count": "PLT",
    "MPV": "MPV",
}


def check_ollama_health() -> dict[str, Any]:
    tags_url = OLLAMA_URL.replace("/api/generate", "/api/tags")

    try:
        response = requests.get(tags_url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise OllamaConnectionError(
            f"Cannot reach Ollama at {tags_url}. Start Ollama and verify the URL."
        ) from exc

    body = response.json()
    models = [m.get("name", "") for m in body.get("models", [])]
    model_found = any(name.startswith(f"{OLLAMA_MODEL}:") or name == OLLAMA_MODEL for name in models)

    if not model_found:
        raise OllamaModelNotFoundError(
            f"Model '{OLLAMA_MODEL}' not found in Ollama. Pull it first."
        )

    return {
        "ok": True,
        "ollama_url": OLLAMA_URL,
        "model": OLLAMA_MODEL,
        "models": models,
    }


def pdf_to_images(pdf_bytes: bytes) -> list[Image.Image]:
    if not pdf_bytes:
        raise ValueError("Empty PDF payload")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(doc) == 0:
        raise ValueError("PDF has no pages")

    images: list[Image.Image] = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap()
        img = Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)

    return images


def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def run_ocr(image: Image.Image) -> str:
    img_b64 = image_to_base64(image)

    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "prompt": "Text Recognition:",
        "images": [img_b64],
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=180)
        if response.status_code >= 400:
            body_preview = response.text[:300].strip()
            raise OllamaConnectionError(
                f"Ollama OCR request failed ({response.status_code}). Response: {body_preview}"
            )

        body = response.json()
        return str(body.get("response", ""))
    except requests.exceptions.RequestException as exc:
        raise OllamaConnectionError(
            f"Failed calling Ollama OCR at {OLLAMA_URL}: {exc.__class__.__name__}: {exc}"
        ) from exc


def _extract_values(text: str) -> dict[str, float]:
    results: dict[str, float] = {}

    for line in text.split("\n"):
        line = line.strip()
        match = re.match(r"(.+?)\s*[:\-]\s*([\d\.]+)", line)
        if not match:
            continue

        name = match.group(1).strip()
        if name not in STANDARD_MAP:
            continue

        try:
            value = float(match.group(2))
        except ValueError:
            continue

        results[STANDARD_MAP[name]] = value

    return results


def extract_cbc_from_pdf(pdf_bytes: bytes) -> dict[str, float]:
    images = pdf_to_images(pdf_bytes)
    final_results: dict[str, float] = {}

    for page_index, image in enumerate(images):
        try:
            ocr_text = run_ocr(image)
            extracted = _extract_values(ocr_text)
            final_results.update(extracted)
        except OllamaConnectionError as exc:
            raise OllamaConnectionError(f"OCR failed on page {page_index + 1}: {exc}") from exc

    if not final_results:
        raise ValueError(
            "Could not extract CBC markers from PDF. Check scan quality or OCR model output format."
        )

    return final_results
