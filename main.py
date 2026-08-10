from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from chart_logic import create_chart_from_csv_bytes


app = FastAPI(
    title="Membership Chart API",
    version="1.0.0",
    description="Upload a CSV and receive the generated membership chart as a PNG.",
)

# For development this allows any frontend to call the API.
# On production you can set CORS_ORIGINS to a comma-separated list:
# https://example.com,https://www.example.com
cors_value = os.getenv("CORS_ORIGINS", "*").strip()

if cors_value == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [
        origin.strip()
        for origin in cors_value.split(",")
        if origin.strip()
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


@app.get("/")
def root():
    return {
        "message": "Membership Chart API is running",
        "upload_endpoint": "/generate-chart",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate-chart")
async def generate_chart(
    file: UploadFile = File(..., description="CSV file used to generate the chart"),
):
    filename = file.filename or "data.csv"

    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=400,
            detail="Please upload a .csv file.",
        )

    csv_bytes = await file.read()

    if not csv_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded CSV is empty.",
        )

    if len(csv_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="CSV is too large. Maximum upload size is 5 MB.",
        )

    try:
        image_buffer = create_chart_from_csv_bytes(csv_bytes)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate chart: {exc}",
        ) from exc

    output_name = f"{Path(filename).stem}_chart.png"

    return StreamingResponse(
        image_buffer,
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="{output_name}"'
        },
    )
