"""
Unit and Integration Tests for MNIST Vision AI Flask Application
Tests cover:
- Homepage HTTP 200 status and UI rendering
- Health endpoint response and schema
- Missing model handling
- Invalid/empty file uploads rejection
- Unsupported image format rejection
- Valid image prediction pipeline execution
- Response schema validation (predicted_digit, confidence, probabilities 0-9)
- Error handling
"""

import io
import json
import pytest
import numpy as np
from PIL import Image

import app as flask_app_module
from app import app, preprocess_digit_image, allowed_file


@pytest.fixture
def client():
    """Configures test client for the Flask application."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_digit_image_bytes():
    """Generates an in-memory test image of a white digit '7' on black background."""
    img = Image.new("L", (28, 28), color=0)
    # Draw simple '7'
    for x in range(6, 22):
        img.putpixel((x, 6), 255)
    for i, y in enumerate(range(7, 24)):
        x = max(6, 21 - (i * 15 // 17))
        img.putpixel((x, y), 255)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_inverted_image_bytes():
    """Generates an in-memory test image of a black digit on white paper."""
    img = Image.new("L", (50, 50), color=255)
    for x in range(15, 35):
        img.putpixel((x, 15), 0)
    for y in range(15, 40):
        img.putpixel((25, y), 0)

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# --- 1. Homepage Tests ---
def test_homepage_status(client):
    """Verifies that the homepage returns HTTP 200 and loads the studio UI."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "MNIST Vision AI" in html
    assert "Handwritten Digit Recognition Studio" in html
    assert "Analyze Digit" in html


# --- 2. Health Endpoint Tests ---
def test_health_endpoint(client):
    """Verifies that /health returns proper status and model info."""
    response = client.get("/health")
    assert response.status_code in (200, 503)
    data = response.get_json()
    assert "status" in data
    assert "model_loaded" in data
    if data["model_loaded"]:
        assert data["status"] == "ok"
        assert data["model_filename"] is not None
        assert data["input_shape"] is not None
        assert data["output_shape"] is not None


# --- 3. Missing Model Handling ---
def test_missing_model_behavior(client, monkeypatch):
    """Verifies that if the model is absent, /health reports degraded and /predict returns 503."""
    # Temporarily set model to None
    original_model = flask_app_module.model
    monkeypatch.setattr(flask_app_module, "model", None)
    monkeypatch.setattr(flask_app_module, "model_load_error", "Mock model missing")
    monkeypatch.setattr(flask_app_module, "load_mnist_model", lambda: False)

    # Health check
    health_resp = client.get("/health")
    assert health_resp.status_code == 503
    health_data = health_resp.get_json()
    assert health_data["status"] == "degraded"
    assert health_data["model_loaded"] is False

    # Predict check
    pred_resp = client.post("/predict", data={"image": (io.BytesIO(b"fake"), "test.png")})
    assert pred_resp.status_code == 503
    pred_data = pred_resp.get_json()
    assert pred_data["success"] is False
    assert "Model is not loaded" in pred_data["error"]


# --- 4. Invalid Uploads & Validations ---
def test_predict_no_file(client):
    """Verifies rejection when no image file is sent in the request."""
    response = client.post("/predict", data={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "No image file provided" in data["error"]


def test_predict_empty_file(client):
    """Verifies rejection of a zero-byte file."""
    data = {"image": (io.BytesIO(b""), "empty.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    res_json = response.get_json()
    assert res_json["success"] is False


def test_predict_corrupt_image(client):
    """Verifies rejection of non-image corrupt bytes."""
    data = {"image": (io.BytesIO(b"this is not a valid image payload"), "corrupt.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    res_json = response.get_json()
    assert res_json["success"] is False
    assert "unable to decode" in res_json["error"].lower()


# --- 5. Allowed Extension Helper ---
def test_allowed_file():
    """Tests file extension validation helper."""
    assert allowed_file("digit.png") is True
    assert allowed_file("test.JPG") is True
    assert allowed_file("scan.jpeg") is True
    assert allowed_file("sample.webp") is True
    assert allowed_file("malicious.exe") is False
    assert allowed_file("script.py") is False
    assert allowed_file("no_extension") is False


# --- 6. Preprocessing Unit Tests ---
def test_preprocessing_pipeline(sample_digit_image_bytes):
    """Tests image preprocessing tensor conversion and dimensions."""
    tensor, processed_pil = preprocess_digit_image(
        image_bytes=sample_digit_image_bytes,
        inversion_mode="no_invert",
        center_digit=True
    )
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 28, 28)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0
    assert processed_pil.size == (28, 28)


def test_preprocessing_auto_inversion(sample_inverted_image_bytes):
    """Tests auto-inversion for dark-on-white images."""
    tensor, processed_pil = preprocess_digit_image(
        image_bytes=sample_inverted_image_bytes,
        inversion_mode="auto",
        center_digit=True
    )
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 28, 28)
    # Corners should be dark (close to 0.0) after inversion
    corner_vals = [tensor[0, 0, 0], tensor[0, 0, 27], tensor[0, 27, 0], tensor[0, 27, 27]]
    assert np.mean(corner_vals) < 0.2


# --- 7. Full Prediction Pipeline & JSON Schema ---
def test_full_prediction_pipeline(client, sample_digit_image_bytes):
    """Verifies real model prediction response adheres to schema."""
    data = {
        "image": (io.BytesIO(sample_digit_image_bytes), "digit7.png"),
        "inversion_mode": "no_invert",
        "center_digit": "true"
    }
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res_json = response.get_json()

    # Schema validation
    assert res_json["success"] is True
    assert isinstance(res_json["predicted_digit"], int)
    assert 0 <= res_json["predicted_digit"] <= 9
    assert isinstance(res_json["confidence"], (int, float))
    assert 0.0 <= res_json["confidence"] <= 100.0

    # Probabilities dictionary check (0 through 9)
    probs = res_json["probabilities"]
    assert isinstance(probs, dict)
    assert len(probs) == 10
    for digit in range(10):
        key = str(digit)
        assert key in probs
        assert isinstance(probs[key], (int, float))
        assert 0.0 <= probs[key] <= 100.0

    # Preprocessed image preview check
    assert "preprocessed_image" in res_json
    assert res_json["preprocessed_image"].startswith("data:image/png;base64,")


# --- 8. Error Handlers ---
def test_404_handler(client):
    """Tests custom 404 handler for API routes."""
    response = client.get("/predict/unknown_route")
    assert response.status_code == 404
    data = response.get_json()
    assert data["success"] is False
    assert "not found" in data["error"].lower()
