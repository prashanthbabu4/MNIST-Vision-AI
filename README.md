# MNIST Vision AI — Handwritten Digit Recognition Studio

A modern, production-grade web application built with **Python Flask, TensorFlow/Keras, HTML5, CSS3, and JavaScript** that classifies handwritten digits (0–9) using deep artificial neural networks (ANN).

---

## 1. Project Overview

**MNIST Vision AI** provides an interactive interface for real-time handwritten digit inference. Users can either drag-and-drop digit images (PNG, JPG, JPEG, WEBP) or draw digits directly onto an interactive canvas pad. The Flask backend processes each image, normalizes and centers the digit using standard MNIST bounding box transformations, and queries the trained Keras model to produce digit predictions, confidence percentages, and class probabilities across all digits from 0 to 9.

---

## 2. Features

- **Dual Input Modes**:
  - **Drag & Drop Upload Zone**: Seamless drag-and-drop file upload with live preview, file format validation, and size limitation checks (up to 5MB).
  - **Interactive Drawing Pad**: Touch- and mouse-compatible sketching canvas with adjustable stroke width and instant clearing.
- **Quick Sample Testing**: One-click preset digits (0, 2, 3, 5, 7, 8) to test the neural network immediately.
- **Intelligent Image Preprocessing**:
  - Automatic EXIF orientation correction.
  - Grayscale conversion (`L` mode).
  - Configurable Background Inversion: Automatic detection of dark-on-white paper scans vs. white-on-dark chalkboards.
  - Bounding Box Centering: Crops the digit, preserves aspect ratio, scales to 20×20, and centers it in a 28×28 canvas (the standard Yann LeCun MNIST specification).
- **Comprehensive Results Visualization**:
  - Large typography hero prediction with dynamic gradient lighting.
  - Confidence percentage badge.
  - Visual side-by-side comparison between the original input and the preprocessed 28×28 model tensor.
  - Animated probability distribution bar chart across all 10 digit classes (0–9).
- **Robust Error Handling**:
  - Structured JSON error responses (no raw stack traces exposed).
  - Graceful handling of missing models and corrupt files.
  - Client-side validation with toast alert notifications.
- **Tested & Production Ready**:
  - Automated test suite with Pytest covering endpoints, error states, and schema compliance.
  - Ready for local serving with Waitress (Windows) or Gunicorn (Linux).

---

## 3. Technology Stack

- **Backend**: Python 3.11–3.13, Flask 3.x, Werkzeug
- **Deep Learning**: TensorFlow 2.x, Keras 3.x
- **Image Processing**: Pillow (PIL), NumPy
- **Frontend**: Vanilla JavaScript (ES6+), HTML5 Canvas, Modern CSS3 (Glassmorphism, CSS Grid & Flexbox, Custom Properties)
- **Production Servers**: Waitress (Windows) / Gunicorn (Linux/Containers)
- **Testing**: Pytest

---

## 4. Project Folder Structure

```text
MNIST_app/
├── app.py                     # Main Flask application & inference routes
├── requirements.txt           # Production & development dependencies
├── README.md                  # Comprehensive documentation
├── .gitignore                 # Git ignore rules
├── .env.example               # Environment variable template
│
├── models/
│   ├── digits_intel.h5        # Trained Keras model (notebook format)
│   └── mnist_model.h5         # Standardized model alias
│
├── scripts/
│   └── train_model.py         # Retraining script replicating the notebook
│
├── templates/
│   └── index.html             # Premium glassmorphic responsive UI
│
├── static/
│   ├── css/
│   │   └── style.css          # Design system & dark mode stylesheet
│   └── js/
│       └── app.js             # Canvas, drag-and-drop, API communication
│
└── tests/
    └── test_app.py            # Automated test suite (Pytest)
```

---

## 5. Model Architecture & Preprocessing

The model is trained according to the notebook `MNIST Digits Classification [0 to 9].ipynb`:

### Model Architecture
- **Layer 1**: `Flatten(input_shape=(28, 28))`
- **Layer 2**: `Dense(100, activation="relu")`
- **Layer 3**: `Dense(30, activation="relu")`
- **Layer 4**: `Dense(10, activation="softmax")`
- **Loss**: `sparse_categorical_crossentropy`
- **Optimizer**: `RMSprop`
- **Test Accuracy**: **97.67%**

### Model Input & Output Expectations
- **Input Shape**: `(batch_size, 28, 28)` single-channel normalized pixel values in the range `[0.0, 1.0]`. White digit strokes (`~1.0`) on black background (`~0.0`).
- **Output Shape**: `(batch_size, 10)` probability vector representing digits `0` through `9`.

---

## 6. Local Setup and Installation

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13 installed.
- Git (optional).

### 2. Create and Activate Virtual Environment

**Windows (PowerShell)**:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Note for Windows users: if your system is missing the Visual C++ Redistributable, ensure `msvcp140.dll` is installed or in your path).*

### 4. Verify Model Placement
The trained model files are located in `models/digits_intel.h5` and `models/mnist_model.h5`. If you wish to retrain or update the model from scratch:
```bash
python scripts/train_model.py
```

---

## 7. Running the Application

### Development Mode
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Production Mode (Windows)
```bash
waitress-serve --port=5000 app:app
```

### Production Mode (Linux / Cloud Container)
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

---

## 8. API Endpoint Documentation

### `GET /`
Renders the single-page application interface.

### `GET /health`
Returns the status of the service and model readiness.

**Sample Response (`200 OK`)**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_filename": "digits_intel.h5",
  "input_shape": [null, 28, 28],
  "output_shape": [null, 10],
  "error": null
}
```

### `POST /predict`
Performs handwritten digit classification.

- **Request Type**: `multipart/form-data`
- **Form Fields**:
  - `image` *(Required)*: File upload (`image/png`, `image/jpeg`, `image/webp`).
  - `inversion_mode` *(Optional)*: `"auto"` (default), `"invert"`, or `"no_invert"`.
  - `center_digit` *(Optional)*: `"true"` (default) or `"false"`.

**Sample Response (`200 OK`)**:
```json
{
  "success": true,
  "predicted_digit": 7,
  "confidence": 99.42,
  "probabilities": {
    "0": 0.01,
    "1": 0.04,
    "2": 0.08,
    "3": 0.12,
    "4": 0.02,
    "5": 0.05,
    "6": 0.01,
    "7": 99.42,
    "8": 0.15,
    "9": 0.10
  },
  "preprocessed_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAY..."
}
```

**Error Response (`400 Bad Request`)**:
```json
{
  "success": false,
  "error": "No image file provided. Please upload an image using the 'image' field."
}
```

---

## 9. Testing Instructions

The project includes an automated Pytest test suite in `tests/test_app.py`.

Run the tests:
```bash
pytest -v
```

Tests verify:
- Homepage HTTP 200 and UI elements.
- `/health` status and metadata.
- Graceful degradation when the model is absent (503 status).
- Validation rejection on missing or empty files.
- Validation rejection on corrupted image data.
- Allowed file extension checking.
- Preprocessing bounding box centering and normalization.
- Full prediction pipeline end-to-end adhering to JSON response schema.
- Custom 404 error handler.

---

## 10. Deployment Guide

### Option A: Render / Railway / Fly.io (Recommended)

1. **Procfile** or Start Command:
   ```text
   web: gunicorn --bind 0.0.0.0:$PORT --workers 2 app:app
   ```
2. **Environment Variables**:
   ```text
   PORT=10000
   FLASK_DEBUG=False
   MODEL_PATH=models/digits_intel.h5
   ```
3. **Repository Setup**:
   Connect your Git repository containing this project. The platform will automatically detect `requirements.txt` and install dependencies.

### Option B: Docker Container

Create a `Dockerfile` in the root directory:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
```

Build and run:
```bash
docker build -t mnist-vision-ai .
docker run -p 5000:5000 mnist-vision-ai
```

---

## 11. Troubleshooting Common Errors

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `ImportError: Could not find MSVCP140.dll` | Missing Visual C++ redistributable on Windows | Install the official [Microsoft Visual C++ Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe). |
| `Model is not loaded` (`503`) | Model file missing from `models/` directory | Run `python scripts/train_model.py` to generate `models/digits_intel.h5`. |
| `File size exceeds 5MB limit` (`413`) | Uploaded image is larger than 5MB | Resize image or increase `MAX_CONTENT_LENGTH` in environment variables. |
| Digit inverted / incorrect prediction | Dark digit on light background not inverted | Set `inversion_mode` to `auto` or `invert` in Advanced Preprocessing options. |
