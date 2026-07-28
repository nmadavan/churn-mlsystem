# Container image for the churn inference API.
# The model is trained during the build so the image is fully self-contained.

FROM python:3.12-slim

# scikit-learn needs the OpenMP runtime (libgomp1) at import time on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached when only source changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code, config, and raw data needed to train and serve.
COPY config.yaml .
COPY src/ ./src/
COPY data/raw/ ./data/raw/

# Train inside the image: produces models/v1/model.pkl and current_best.json.
# Training falls back to raw data when no ingested training table is present.
RUN python -m src.train

EXPOSE 8000

# Liveness check using the stdlib (no curl on slim).
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
