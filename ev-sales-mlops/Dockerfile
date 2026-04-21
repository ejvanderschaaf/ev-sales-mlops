# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /opt/ml

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY src/       src/
COPY inference/ inference/

ENV PYTHONPATH=/opt/ml
ENV MODEL_PATH=/opt/ml/model/model.pkl

# SageMaker requires the inference server on port 8080
EXPOSE 8080

# Default: run inference server
# Override CMD to run training: docker run ... python src/train_and_tune.py
CMD ["python", "inference/predict.py"]
