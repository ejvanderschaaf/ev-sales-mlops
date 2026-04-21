FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

WORKDIR /opt/ml

COPY --from=builder /install /usr/local
COPY src/       src/
COPY inference/ inference/

ENV PYTHONPATH=/opt/ml
ENV MODEL_PATH=/opt/ml/model/model.pkl

RUN chmod +x /opt/ml/inference/serve
ENV PATH="/opt/ml/inference:${PATH}"

EXPOSE 8080

CMD ["serve"]
