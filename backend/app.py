import os
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import io

from rules import evaluate_sample, THRESHOLDS
from summary import generate_summary
from synthetic_data import generate_synthetic_batch
from clustering import cluster_failures
from export import result_to_xlsx_bytes, result_to_csv_bytes

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow requests from your frontend

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_biscuit_model.keras")
IMG_SIZE = (224, 224)

CLASS_NAMES = ["Cant be eaten", "Can be eaten"]

# ─── Load Model (once at startup, not per request) ────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded! Input shape: {model.input_shape}, Output shape: {model.output_shape}")


# ─── Helper: Preprocess Image ─────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Biscuit Classification API",
        "endpoints": {
            "POST /predict":      "Upload an image to classify (visual inspection)",
            "POST /predict-lab":  "Send lab parameter values (JSON) to check against SNI 2973:2011 thresholds",
            "GET  /thresholds":   "List all lab parameters this API knows how to evaluate, with their limits",
            "GET  /clusters":     "Generate a synthetic batch of samples and cluster the failures by pattern (demo only — see synthetic_data.py)",
            "GET  /clusters/export": "Same as /clusters but downloads as .xlsx (default) or .csv — visit in a browser to trigger a file download",
            "GET  /health":       "Check if the API is running"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file found. Send a file with key 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    image_bytes = file.read()

    try:
        processed = preprocess_image(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400

    raw_output = model.predict(processed, verbose=0)
    confidence = float(raw_output[0][0])

    if confidence >= 0.5:
        predicted_class = CLASS_NAMES[1]
        class_confidence = confidence
    else:
        predicted_class = CLASS_NAMES[0]
        class_confidence = 1.0 - confidence

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": round(class_confidence * 100, 2),
        "raw_output": round(confidence, 6)
    })


@app.route("/thresholds", methods=["GET"])
def thresholds():
    return jsonify(THRESHOLDS)


@app.route("/predict-lab", methods=["POST"])
def predict_lab():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    if not isinstance(data, dict) or len(data) == 0:
        return jsonify({"error": "Send at least one lab parameter, e.g. {\"moisture\": 6.2}"}), 400

    for key, value in data.items():
        if not isinstance(value, (int, float)):
            return jsonify({"error": f"'{key}' must be a number, got: {value!r}"}), 400

    result = evaluate_sample(data)
    summary_text = generate_summary(data, result)

    return jsonify({
        "input": data,
        "passed": result["passed"],
        "failures": result["failures"],
        "failure_categories": result["failure_categories"],
        "summary": summary_text,
    })


def _run_clustering(request_args) -> dict:
    """Shared by /clusters and /clusters/export so both read the same
    query params the same way and never drift out of sync."""
    n = request_args.get("n", default=300, type=int)
    fail_ratio = request_args.get("fail_ratio", default=0.35, type=float)
    n_clusters = request_args.get("n_clusters", default=3, type=int)

    n = max(10, min(n, 2000))
    n_clusters = max(2, min(n_clusters, 6))

    batch = generate_synthetic_batch(n=n, fail_ratio=fail_ratio)
    result = cluster_failures(batch, n_clusters=n_clusters)
    result["data_source"] = "synthetic"
    return result


@app.route("/clusters", methods=["GET"])
def clusters():
    """
    Demo-only endpoint: generates a synthetic batch of lab samples (see
    synthetic_data.py — grounded in our real thresholds, not arbitrary
    random noise), runs every sample through the existing rule engine,
    and clusters the FAILED samples into groups using KMeans.

    This exists to satisfy the case brief's "identify product quality
    failure patterns" / "grouping samples" requirement, since a pure
    rule engine alone only classifies (pass/fail) and doesn't discover
    patterns across samples.

    Query params (all optional):
        n            - how many synthetic samples to generate (default 300)
        fail_ratio   - roughly what fraction should fail (default 0.35)
        n_clusters   - how many clusters to form (default 3)

    IMPORTANT: the underlying data is synthetic. Say so in the demo video.
    """
    return jsonify(_run_clustering(request.args))


@app.route("/clusters/export", methods=["GET"])
def clusters_export():
    """
    Same data as GET /clusters, but returned as an actual downloadable
    file instead of raw JSON — so it opens directly in Excel/Sheets.

    Query params: same as /clusters, plus:
        format - "xlsx" (default) or "csv"

    Visiting this URL directly in a browser (not just curl/Postman)
    triggers a normal file download, since we set Content-Disposition
    below.
    """
    result = _run_clustering(request.args)
    fmt = request.args.get("format", default="xlsx").lower()

    if not result["clusters"]:
        return jsonify({"error": result.get("warning", "No clusters to export.")}), 400

    if fmt == "csv":
        file_bytes = result_to_csv_bytes(result)
        return send_file(
            io.BytesIO(file_bytes),
            mimetype="text/csv",
            as_attachment=True,
            download_name="biscuit_clusters.csv",
        )

    file_bytes = result_to_xlsx_bytes(result)
    return send_file(
        io.BytesIO(file_bytes),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name="biscuit_clusters.xlsx",
    )


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
