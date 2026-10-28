# Production-Grade Prompt: MNIST Prediction Web App Using Flask

Act as a senior full-stack engineer, machine learning engineer, UI/UX designer, and software architect.

I have already trained and uploaded my MNIST model file. I want you to build a complete, production-quality MNIST web application using **Python Flask, TensorFlow/Keras, HTML, CSS, and JavaScript**.

The application must allow users to upload an image of a handwritten digit, predict the digit from 0–9 using my trained model, view prediction confidence, clear the previous prediction, and upload another image.

Do not give me only a tutorial or code snippets. Create the complete working project with all required files, a proper folder structure, and step-by-step instructions to run and deploy it.

## 1. Project Objectives

Build a beautiful, responsive web application with the following features:

- Upload handwritten digit images in PNG, JPG, JPEG, or WEBP format.
- Predict the handwritten digit using my existing trained MNIST model.
- Display the uploaded image and predicted digit side by side.
- Display prediction confidence as a percentage.
- Display probabilities for all 10 digits (0–9) if the model output supports this.
- Provide a Clear / Reset button to remove the uploaded image and prediction.
- Provide an Upload Another Image button.
- Include loading animations while the model is processing an image.
- Display friendly validation and error messages.
- Support multiple predictions without refreshing the entire page.
- Make the interface responsive on desktop, tablet, and mobile.

## 2. Technology Stack

Use the following technologies:

- Frontend: HTML5, CSS3, JavaScript.
- Backend: Python Flask.
- Machine learning: TensorFlow/Keras.
- Image processing: Pillow (PIL) and NumPy.
- Model format: Use my uploaded trained model file.
- Deployment: Choose a suitable Python-hosting platform that supports Flask and TensorFlow.

Do not use a frontend-only mock prediction. Every prediction must come from the actual trained model through the Flask backend.

## 3. Model Integration

First, inspect the project directory and identify my uploaded model file.

Use the actual model filename and do not rename it unnecessarily. If the model is named `mnist_model.h5`, use that filename; otherwise, use the real uploaded filename.

Load the model once when the Flask application starts, rather than reloading it for every prediction.

Example:

```python
import tensorflow as tf
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "mnist_model.h5"

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)
```

Adapt the path and model-loading code to my actual file and model format.

Before implementing preprocessing, inspect the model's input shape, output shape, and expected data format.

Important: Do not assume that every MNIST model has the same architecture or output format.

## 4. Image Preprocessing

Implement reliable preprocessing for uploaded images:

1. Validate the file type and file size.
2. Open the uploaded image safely using Pillow.
3. Convert the image to grayscale.
4. Correct image orientation when applicable.
5. Resize or crop the digit according to the model's expected input dimensions, usually 28 × 28 pixels.
6. Normalize pixel values according to the model's training preprocessing.
7. Reshape the image to match the model's expected input shape.
8. Run inference using the trained model.

MNIST images commonly use white digits on a black background, but uploaded images may have the opposite colors.

Provide appropriate preprocessing for both white-on-black and black-on-white images. Use a configurable inversion option or a carefully designed automatic approach. Do not blindly invert every image.

If useful, detect the digit's bounding box and center the digit while preserving its aspect ratio.

Do not assume the model was trained with a particular normalization or image inversion strategy. Document the preprocessing assumptions and make them configurable when necessary.

## 5. Prediction Results

Create an attractive results section that displays:

- Uploaded image preview.
- Predicted digit in large typography.
- Confidence percentage.
- Probability distribution for digits 0–9, if applicable.
- A visual bar chart for the probability distribution.
- Prediction processing status.

If the model outputs 10-class probabilities, display them directly.

If the model outputs logits, apply the appropriate softmax transformation. If the model outputs only a digit or a reconstruction rather than class probabilities, handle that case correctly instead of inventing confidence scores.

Display a helpful message if the model is not a compatible 10-class classifier.

## 6. Premium UI/UX Design

Create a modern, premium interface inspired by professional AI products.

Design direction:

- Deep navy background.
- Elegant indigo and violet gradients.
- Electric blue accent colors.
- Soft glassmorphism cards.
- Subtle shadows and borders.
- Rounded corners.
- Clean typography.
- Smooth hover transitions.
- Minimal, polished animations.
- Clear visual hierarchy.
- Excellent spacing and alignment.

Suggested color palette:

- Background: `#0B1020`
- Card background: `#121A2F`
- Primary accent: `#6366F1`
- Secondary accent: `#8B5CF6`
- Highlight: `#38BDF8`
- Main text: `#F8FAFC`
- Secondary text: `#94A3B8`

### Homepage layout

**Header**
- App logo using a brain or neural-network icon.
- App name: `MNIST Vision AI`
- Subtitle: `Handwritten Digit Recognition Studio`
- Small badge: `Powered by TensorFlow`

**Hero section**
- Main heading: `Turn Handwritten Digits Into AI Predictions`
- Supporting description explaining the application.
- Attractive but lightweight visual treatment.

**Upload section**
- Large drag-and-drop upload area.
- Upload icon.
- Text: `Drag & drop your digit image here`
- Supported file format information.
- Browse Files button.
- Optional preview before prediction.

**Prediction button**
- Label: `Analyze Digit`
- Premium gradient styling.
- Loading spinner while processing.
- Disabled state when no valid image is selected.

**Results section**
- Original image preview.
- Predicted digit displayed prominently.
- Confidence percentage.
- Probability chart for digits 0–9.
- Clear and Upload Another Image buttons.

**Footer**
- `Built with Flask, TensorFlow & Python`
- Simple copyright notice without inventing a company or organization.

Do not overcrowd the interface. Keep it visually impressive, accessible, and easy to use.

## 7. Backend Requirements

Create a Flask backend with the following routes:

- `GET /` — render the homepage.
- `POST /predict` — receive an image, preprocess it, run inference, and return prediction results as JSON.
- `GET /health` — return a basic health status and model-loading status.

The prediction API should return a structured response similar to:

```json
{
  "success": true,
  "predicted_digit": 7,
  "confidence": 98.5,
  "probabilities": {
    "0": 0.1,
    "1": 0.2,
    "2": 0.1,
    "3": 0.2,
    "4": 0.1,
    "5": 0.2,
    "6": 0.1,
    "7": 98.5,
    "8": 0.5,
    "9": 0.5
  }
}
```

This is an illustrative response format only. Calculate actual values from the model and ensure the probability values use a consistent scale.

Implement:

- Input validation.
- File-size limits.
- Safe error handling.
- Correct HTTP status codes.
- JSON responses for API errors.
- Secure handling of uploaded files.
- No unnecessary permanent storage of user-uploaded images.
- A configurable upload-size limit.
- Appropriate logging without exposing sensitive information.
- Model initialization and inference error handling.

Never expose Python stack traces or internal filesystem paths to end users.

## 8. Frontend Functionality

Use JavaScript `fetch()` to send the uploaded image to `/predict`.

Implement:

- Drag-and-drop upload.
- File picker.
- Image preview.
- Image format and size validation.
- Analyze button.
- Loading state.
- Prediction result rendering.
- Probability chart rendering.
- Clear/reset functionality.
- Upload another image.
- Friendly error notifications.
- Keyboard accessibility and visible focus states.

The page must update dynamically without a full-page reload when a prediction is completed.

Prevent duplicate submissions while inference is running.

## 9. Required Project Folder Structure

Create a clean structure similar to this:

```text
mnist-vision-ai/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── models/
│   └── mnist_model.h5
│
├── templates/
│   └── index.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   ├── js/
│   │   └── app.js
│   │
│   └── images/
│
└── tests/
    └── test_app.py
```

Adapt the model filename and directory to the actual uploaded model.

Do not create unnecessary files. Include all files required for a functioning application.

## 10. Dependencies

Create a compatible `requirements.txt` containing the dependencies required by the actual implementation.

Potential dependencies include:

- Flask
- TensorFlow
- NumPy
- Pillow
- Gunicorn for production hosting
- Pytest for testing

Choose mutually compatible versions based on the Python version and hosting environment.

Do not blindly install the latest version of every dependency. Avoid unnecessary packages.

## 11. Security and Production Readiness

Implement appropriate protections:

- Maximum upload size.
- File extension and decoded-image validation.
- Safe image decoding.
- No execution of uploaded content.
- No hardcoded secrets.
- Controlled error messages.
- Correct relative paths.
- Model-loading validation.
- Appropriate cache headers where necessary.
- Production server configuration.
- Basic automated tests for the homepage, health endpoint, invalid uploads, and prediction behavior.

Keep the implementation simple enough for a beginner to maintain while following sound production practices.

## 12. Deployment Instructions

Provide exact deployment instructions for a platform that supports Python, Flask, and TensorFlow.

Include:

1. How to arrange the project files.
2. How to create and activate a Python virtual environment.
3. How to install dependencies.
4. How to run the app locally.
5. How to test predictions.
6. How to configure the deployment platform.
7. How to configure the production start command.
8. How to deploy the application.
9. How to check deployment logs.
10. How to troubleshoot common deployment errors.

Important: Do not recommend deploying the complete Flask/TensorFlow backend as a static-only Netlify site. If Netlify is desired, explain that a separate Python-capable backend or a suitable serverless inference architecture is required.

Include an appropriate production start command, for example:

```bash
gunicorn app:app
```

Adjust it if the application entry point differs.

## 13. Testing Requirements

Create tests for:

- Homepage returns HTTP 200.
- Health endpoint works.
- Missing model produces a clear startup error.
- Invalid file uploads are rejected.
- Unsupported image formats are rejected.
- Valid images reach the prediction pipeline.
- Prediction responses follow the documented JSON schema.
- Model inference errors are handled safely.

Use a mock model in unit tests where appropriate so basic tests do not require loading the full trained model.

## 14. README Documentation

Write a complete README that explains:

- Project overview.
- Features.
- Technology stack.
- Folder structure.
- Model file placement.
- Model input and output expectations.
- Local setup.
- Run commands.
- API endpoint documentation.
- Testing instructions.
- Deployment instructions.
- Common errors and solutions.

## 15. How You Must Work

Follow this order:

1. Inspect the uploaded model and project files.
2. Determine whether the model is a classifier, autoencoder, or another model type.
3. Identify its input and output shapes.
4. Confirm the correct preprocessing pipeline.
5. Generate the complete project structure.
6. Implement the Flask backend.
7. Implement the complete frontend.
8. Add validation, error handling, and tests.
9. Verify that the application uses real model inference.
10. Provide exact run and deployment instructions.

If you cannot directly inspect the model, clearly state what information is missing and provide a small diagnostic script to inspect it.

Do not silently replace the trained model with a pretrained model, dummy predictions, or hardcoded results.

Do not omit code using phrases such as “implement the remaining logic yourself.”

**Final deliverables:** Complete runnable source code for every required file, the correct model integration, a premium responsive UI, test instructions, a deployment guide, and troubleshooting steps.

Start by inspecting my uploaded model and project files. Then build the complete application.