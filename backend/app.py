import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
import io

from rules import evaluate_sample, THRESHOLDS
from summary import generate_summary

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow requests from your frontend

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_biscuit_model.keras")
IMG_SIZE = (224, 224)

# TODO: Replace these with your actual class names!
# Since your model has output shape (1,) it's BINARY classification.
# Class 0 = below 0.5 threshold, Class 1 = above 0.5 threshold
CLASS_NAMES = ["Cant be eaten", "Can be eaten"]

# ─── Load Model (once at startup, not per request) ────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Model loaded! Input shape: {model.input_shape}, Output shape: {model.output_shape}")


# ─── Helper: Preprocess Image ─────────────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Takes raw image bytes, returns a preprocessed numpy array
    ready to be fed into the model.

    Steps:
    1. Open image from bytes
    2. Convert to RGB (handles RGBA, grayscale, etc.)
    3. Resize to 224x224
    4. Convert to numpy array
    5. Apply MobileNetV2 preprocessing (scales pixels to [-1, 1])
    6. Add batch dimension: shape becomes (1, 224, 224, 3)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # (224,224,3) -> (1,224,224,3)
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
            "GET  /health":       "Check if the API is running"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    # 1. Validate: make sure an image file was sent
    if "image" not in request.files:
        return jsonify({"error": "No image file found. Send a file with key 'image'"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # 2. Read image bytes
    image_bytes = file.read()

    # 3. Preprocess
    try:
        processed = preprocess_image(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400

    # 4. Run model inference
    raw_output = model.predict(processed, verbose=0)  # shape: (1, 1)
    confidence = float(raw_output[0][0])              # scalar between 0 and 1

    # 5. Determine class
    # Since output is (1,) with sigmoid:
    #   confidence >= 0.5  → Class 1
    #   confidence <  0.5  → Class 0
    if confidence >= 0.5:
        predicted_class = CLASS_NAMES[1]
        class_confidence = confidence
    else:
        predicted_class = CLASS_NAMES[0]
        class_confidence = 1.0 - confidence  # flip so confidence always = how sure we are

    # 6. Return result
    return jsonify({
        "predicted_class": predicted_class,
        "confidence": round(class_confidence * 100, 2),  # e.g. 87.43
        "raw_output": round(confidence, 6)               # raw sigmoid value, useful for debugging
    })


@app.route("/thresholds", methods=["GET"])
def thresholds():
    """
    Lets the frontend build the Predictive Simulator form dynamically
    instead of hardcoding parameter names/limits in HTML. If you add a new
    parameter to rules.THRESHOLDS later, the form updates automatically —
    no frontend code changes needed.
    """
    return jsonify(THRESHOLDS)


@app.route("/predict-lab", methods=["POST"])
def predict_lab():
    """
    Rule-based lab quality check (the "Predictive Simulator" backend).

    Expects JSON body, e.g.:
        {"moisture": 6.2, "protein": 6.0, "ash": 0.8}

    You don't need to send every parameter — only the ones you want
    evaluated. This lets a user test "what if just moisture changes?"
    without filling in every field.
    """
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    if not isinstance(data, dict) or len(data) == 0:
        return jsonify({"error": "Send at least one lab parameter, e.g. {\"moisture\": 6.2}"}), 400

    # Validate every value is actually a number before we do math on it —
    # otherwise a stray string like "6.2%" would crash the comparison
    # inside evaluate_sample() instead of returning a clean error.
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


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
