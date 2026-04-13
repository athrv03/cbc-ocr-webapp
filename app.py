from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

try:
    from .ocr import (
        OllamaConnectionError,
        OllamaModelNotFoundError,
        check_ollama_health,
        extract_cbc_from_pdf,
    )
    from .predictor import get_predictor, predict_patient
except ImportError:
    from ocr import (
        OllamaConnectionError,
        OllamaModelNotFoundError,
        check_ollama_health,
        extract_cbc_from_pdf,
    )
    from predictor import get_predictor, predict_patient

app = FastAPI(title="CBC OCR + XGB Web App", version="1.0.0")

base_dir = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(base_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/ocr-health")
def ocr_health() -> dict[str, Any]:
    try:
        return check_ollama_health()
    except OllamaConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OllamaModelNotFoundError as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc


@app.post("/api/predict-from-pdf")
async def predict_from_pdf(file: UploadFile = File(...)) -> dict[str, Any]:
    try:
        if not file.filename:
            raise ValueError("Missing file name")
        if not file.filename.lower().endswith(".pdf"):
            raise ValueError("Please upload a PDF file")

        pdf_bytes = await file.read()
        extracted_values = extract_cbc_from_pdf(pdf_bytes)
        result_df = predict_patient(extracted_values)
        predicted_labels = result_df.loc[result_df["Predicted"], "Disease"].tolist()

        return {
            "file_name": file.filename,
            "extracted_values": extracted_values,
            "predicted_diseases": predicted_labels if predicted_labels else ["Normal"],
            "results": result_df.to_dict(orient="records"),
        }
    except OllamaConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except OllamaModelNotFoundError as exc:
        raise HTTPException(status_code=424, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Eager load model artifacts at startup
get_predictor()
