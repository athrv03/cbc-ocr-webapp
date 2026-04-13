# CBC OCR Full-Stack Web App

Upload a lab PDF -> run `glm-ocr` through Ollama -> map values to CBC JSON -> run XGBoost prediction -> show results in a web UI.

## Prerequisites

- Python 3.11+
- Ollama running locally
- `glm-ocr` model available in Ollama
- Existing model artifacts in parent folder:
  - `../modeldir/xgb_disease_model_v1.pkl`
  - `../modeldir/label_binarizer.pkl`
  - `../modeldir/feature_columns.pkl`
  - `../cbc_dataframe_with_disease.csv`

## Setup

```bash
cd "cbc_ocr_webapp"
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

(Optional env variables)

```bash
export OLLAMA_URL="http://localhost:11434/api/generate"
export OLLAMA_MODEL="glm-ocr"
```

## Run

```bash
./.venv/bin/uvicorn app:app --reload
```

Open:

- `http://127.0.0.1:8000`

## API

- `POST /api/predict-from-pdf` (multipart file upload)
- `GET /api/health`
