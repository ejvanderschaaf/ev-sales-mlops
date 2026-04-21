import logging
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# SageMaker mounts the model artifact at /opt/ml/model/
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/ml/model/model.pkl")
_model = None


def load_model():
    global _model
    if _model is None:
        logger.info(f"Loading model from {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
    return _model


@app.route("/ping", methods=["GET"])
def ping():
    """SageMaker health check endpoint."""
    try:
        load_model()
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/invocations", methods=["POST"])
def invocations():
    """SageMaker inference endpoint. Accepts JSON list of feature dicts."""
    try:
        model = load_model()
        data = request.get_json(force=True)

        # Accept single dict or list of dicts
        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)
        predictions = model.predict(df)

        return jsonify({"predictions": predictions.tolist()}), 200

    except Exception as e:
        logger.error(f"Inference error: {e}")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # Local dev: python inference/predict.py
    # SageMaker: container entrypoint calls this directly
    app.run(host="0.0.0.0", port=8080)
