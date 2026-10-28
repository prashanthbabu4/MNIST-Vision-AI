"""
MNIST Vision AI — Production Flask Backend
Serves the digit recognition studio, handles image uploads and drawing canvas submissions,
preprocesses digit imagery (orientation, grayscale, inversion, bounding box centering),
and performs real-time inference using the trained Keras/TensorFlow model.
"""

import os
import io
import sys
import base64
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps
import numpy as np

# --- Windows DLL Directory Setup ---
BASE_DIR = Path(__file__).resolve().parent
venv_scripts = BASE_DIR / ".venv" / "Scripts"
if venv_scripts.exists() and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(str(venv_scripts))
    except Exception:
        pass

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("mnist_vision_ai")

# --- Flask Initialization & Configuration ---
app = Flask(__name__)

# Maximum upload size (5MB default, configurable via environment)
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 5 * 1024 * 1024))
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

# --- Model Loading & Management ---
MODEL_CANDIDATE_PATHS = [
    os.environ.get("MODEL_PATH"),
    BASE_DIR / "models" / "digits_intel.h5",
    BASE_DIR / "models" / "mnist_model.h5",
    BASE_DIR / "digits_intel.h5",
    BASE_DIR / "mnist_model.h5"
]

model = None
model_filename = None
model_input_shape = None
model_output_shape = None
model_load_error = None


def load_mnist_model() -> bool:
    """
    Locates and loads the trained Keras/TensorFlow model once at application startup.
    Supports .h5, .keras, or SavedModel directory formats.
    """
    global model, model_filename, model_input_shape, model_output_shape, model_load_error

    selected_path = None
    for candidate in MODEL_CANDIDATE_PATHS:
        if candidate and Path(candidate).exists():
            selected_path = Path(candidate)
            break

    if not selected_path:
        model_load_error = (
            f"No model file found in candidate paths: {[str(p) for p in MODEL_CANDIDATE_PATHS if p]}"
        )
        logger.warning(model_load_error)
        return False

    try:
        import keras

        logger.info(f"Loading trained MNIST model from '{selected_path}'...")
        # Compile=False ensures inference works cleanly even if custom metrics/losses were defined
        model = keras.models.load_model(str(selected_path), compile=False)
        model_filename = selected_path.name

        # Extract shapes
        model_input_shape = list(model.input_shape) if hasattr(model, "input_shape") else None
        model_output_shape = list(model.output_shape) if hasattr(model, "output_shape") else None

        logger.info(f"Model '{model_filename}' loaded successfully!")
        logger.info(f"Input Shape: {model_input_shape}, Output Shape: {model_output_shape}")
        model_load_error = None
        return True
    except Exception as exc:
        model_load_error = f"Failed to load model from '{selected_path}': {str(exc)}"
        logger.error(model_load_error, exc_info=True)
        model = None
        return False


# Attempt loading on module import
load_mnist_model()


# --- Image Preprocessing Helpers ---
def allowed_file(filename: str) -> bool:
    """Verifies that the filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_digit_image(
    image_bytes: bytes,
    inversion_mode: str = "auto",
    center_digit: bool = True
) -> Tuple[np.ndarray, Image.Image]:
    """
    Preprocesses an input image according to the MNIST requirements:
    1. Opens and validates image bytes.
    2. Corrects EXIF orientation if needed.
    3. Converts to single-channel grayscale ('L').
    4. Evaluates background brightness and handles inversion (MNIST digits are white on black).
    5. Bounding box cropping and aspect-ratio preserved centering into 20x20 on 28x28 canvas.
    6. Normalizes pixel values to [0.0, 1.0].
    7. Reshapes to match model expected input tensor.

    Returns:
        (input_tensor, preprocessed_pil_image)
    """
    # 1. Open safely
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:
        raise ValueError("Invalid image file: unable to decode image data.") from exc

    # 2. Correct EXIF orientation
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass

    # 3. Convert to grayscale
    gray_img = pil_img.convert("L")

    # 4. Inversion logic
    # MNIST standard: digit strokes are white (~255) on black background (~0).
    # Uploaded photos/scans typically have black ink on white paper.
    img_np = np.array(gray_img, dtype=np.uint8)

    # Border mean brightness test (average of top, bottom, left, right border bands)
    h, w = img_np.shape
    border_pixels = np.concatenate([
        img_np[0:max(1, h // 10), :].flatten(),
        img_np[-max(1, h // 10):, :].flatten(),
        img_np[:, 0:max(1, w // 10)].flatten(),
        img_np[:, -max(1, w // 10):].flatten()
    ])
    border_brightness = float(np.mean(border_pixels))

    should_invert = False
    if inversion_mode == "invert":
        should_invert = True
    elif inversion_mode == "no_invert":
        should_invert = False
    else:  # 'auto'
        # If border is light (> 127), it's dark ink on light background -> invert to white on dark
        should_invert = border_brightness > 127

    if should_invert:
        gray_img = ImageOps.invert(gray_img)
        img_np = np.array(gray_img, dtype=np.uint8)

    # 5. Centering & aspect-ratio normalization (MNIST standard bounding box centering)
    if center_digit:
        # Threshold to identify digit pixels (strokes)
        threshold = max(20, int(np.percentile(img_np, 75) * 0.35))
        mask = img_np > threshold
        coords = np.argwhere(mask)

        if len(coords) > 10:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # Crop digit bounding box
            digit_crop = gray_img.crop((x_min, y_min, x_max + 1, y_max + 1))
            dw, dh = digit_crop.size

            # Scale digit so its largest dimension is 20 pixels
            if dw > dh:
                new_w = 20
                new_h = max(1, int(round((20.0 / dw) * dh)))
            else:
                new_h = 20
                new_w = max(1, int(round((20.0 / dh) * dw)))

            digit_resized = digit_crop.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # Create black 28x28 canvas
            canvas = Image.new("L", (28, 28), color=0)
            # Center in 28x28
            paste_x = (28 - new_w) // 2
            paste_y = (28 - new_h) // 2
            canvas.paste(digit_resized, (paste_x, paste_y))
            final_img = canvas
        else:
            # Blank or near-blank image; direct resize
            final_img = gray_img.resize((28, 28), Image.Resampling.LANCZOS)
    else:
        final_img = gray_img.resize((28, 28), Image.Resampling.LANCZOS)

    # 6. Normalize pixel values [0.0, 1.0] (matching notebook: image = image / 255.0)
    norm_array = np.array(final_img, dtype=np.float32) / 255.0

    # 7. Reshape according to model expectations
    # In notebook: model has Flatten() input, predicts on (1, 28, 28)
    if model_input_shape and len(model_input_shape) == 2:
        # Dense input (e.g., None, 784)
        input_tensor = norm_array.reshape(1, 784)
    elif model_input_shape and len(model_input_shape) == 4 and model_input_shape[-1] == 1:
        # CNN input (e.g., None, 28, 28, 1)
        input_tensor = np.expand_dims(norm_array, axis=(0, -1))
    else:
        # Default notebook format (None, 28, 28)
        input_tensor = np.expand_dims(norm_array, axis=0)

    return input_tensor, final_img


def encode_image_base64(image: Image.Image) -> str:
    """Encodes a PIL Image into a base64 PNG data URL."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"


# --- Routes ---
@app.route("/", methods=["GET"])
def index():
    """Renders the MNIST Vision AI web interface."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Returns application and model health status."""
    is_loaded = model is not None
    return jsonify({
        "status": "ok" if is_loaded else "degraded",
        "model_loaded": is_loaded,
        "model_filename": model_filename,
        "input_shape": model_input_shape,
        "output_shape": model_output_shape,
        "error": model_load_error
    }), (200 if is_loaded else 503)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receives an image file, preprocesses it, performs inference with the trained model,
    and returns predicted digit, confidence percentage, and full probability distribution.
    """
    global model
    if model is None:
        # Try reloading once in case model was saved after startup
        if not load_mnist_model():
            return jsonify({
                "success": false if False else False,
                "error": "Model is not loaded. Please ensure the model file exists.",
                "details": model_load_error
            }), 503

    # Check for image file in request
    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided. Please upload an image using the 'image' field."
        }), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No file selected."
        }), 400

    # Optional parameters
    inversion_mode = request.form.get("inversion_mode", "auto").strip().lower()
    center_digit = request.form.get("center_digit", "true").strip().lower() in ("true", "1", "yes")

    try:
        image_bytes = file.read()
        if len(image_bytes) == 0:
            return jsonify({
                "success": False,
                "error": "The uploaded file is empty."
            }), 400

        # Preprocess
        input_tensor, processed_img = preprocess_digit_image(
            image_bytes=image_bytes,
            inversion_mode=inversion_mode,
            center_digit=center_digit
        )

        # Run model inference
        raw_predictions = model.predict(input_tensor, verbose=0)

        # Check if output is logits or softmax
        preds_row = raw_predictions[0]
        # If output does not sum close to 1 or contains negative numbers, apply softmax
        if np.any(preds_row < 0) or not np.isclose(np.sum(preds_row), 1.0, atol=0.05):
            exp_preds = np.exp(preds_row - np.max(preds_row))
            probs = exp_preds / np.sum(exp_preds)
        else:
            probs = preds_row

        predicted_digit = int(np.argmax(probs))
        confidence = float(np.max(probs) * 100.0)

        # Build probabilities dictionary { "0": float, ..., "9": float }
        probabilities: Dict[str, float] = {}
        for idx in range(min(10, len(probs))):
            probabilities[str(idx)] = round(float(probs[idx] * 100.0), 2)

        # Encode preprocessed 28x28 image for frontend visual comparison
        preprocessed_b64 = encode_image_base64(processed_img)

        return jsonify({
            "success": True,
            "predicted_digit": predicted_digit,
            "confidence": round(confidence, 2),
            "probabilities": probabilities,
            "preprocessed_image": preprocessed_b64
        }), 200

    except ValueError as val_err:
        logger.warning(f"Validation error: {val_err}")
        return jsonify({
            "success": False,
            "error": str(val_err)
        }), 400
    except Exception as exc:
        logger.error(f"Inference error: {exc}", exc_info=True)
        return jsonify({
            "success": False,
            "error": "An internal error occurred during prediction. Please try again."
        }), 500


# --- Error Handlers ---
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "error": f"Uploaded file exceeds the maximum allowed size of {MAX_CONTENT_LENGTH // (1024 * 1024)}MB."
    }), 413


@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "success": False,
        "error": "Bad request. Please verify the request parameters."
    }), 400


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/predict") or request.path.startswith("/health"):
        return jsonify({
            "success": False,
            "error": "Endpoint not found."
        }), 404
    return render_template("index.html"), 404


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error occurred."
    }), 500


# --- Entry Point ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1")
    logger.info(f"Starting MNIST Vision AI server on http://{host}:{port} (debug={debug})...")
    app.run(host=host, port=port, debug=debug)
