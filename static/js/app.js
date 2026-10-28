/**
 * MNIST Vision AI — Client Application
 * Handles file uploads, drag & drop, interactive canvas, real-time API communication,
 * result rendering, and probability chart animations.
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- State ---
    let currentInputMode = 'upload'; // 'upload' | 'draw'
    let selectedFile = null;
    let isAnalyzing = false;
    let hasDrawn = false;

    // --- DOM Elements ---
    const healthPill = document.getElementById('healthPill');
    const healthStatusText = document.getElementById('healthStatusText');

    const tabUploadBtn = document.getElementById('tabUploadBtn');
    const tabDrawBtn = document.getElementById('tabDrawBtn');
    const uploadTabPane = document.getElementById('uploadTabPane');
    const drawTabPane = document.getElementById('drawTabPane');

    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const dropzoneEmpty = document.getElementById('dropzoneEmpty');
    const dropzonePreview = document.getElementById('dropzonePreview');
    const previewImage = document.getElementById('previewImage');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const removeFileBtn = document.getElementById('removeFileBtn');

    const drawingCanvas = document.getElementById('drawingCanvas');
    const brushSizeInput = document.getElementById('brushSize');
    const brushSizeVal = document.getElementById('brushSizeVal');
    const clearCanvasBtn = document.getElementById('clearCanvasBtn');

    const toggleOptionsBtn = document.getElementById('toggleOptionsBtn');
    const optionsContent = document.getElementById('optionsContent');
    const inversionModeSelect = document.getElementById('inversionMode');
    const centerDigitCheck = document.getElementById('centerDigitCheck');

    const analyzeBtn = document.getElementById('analyzeBtn');
    const analyzeBtnText = document.getElementById('analyzeBtnText');
    const analyzeSpinner = document.getElementById('analyzeSpinner');
    const analyzeIcon = document.getElementById('analyzeIcon');

    const resultsCard = document.getElementById('resultsCard');
    const resultsEmpty = document.getElementById('resultsEmpty');
    const resultsBody = document.getElementById('resultsBody');
    const clearPredictionBtn = document.getElementById('clearPredictionBtn');
    const uploadAnotherBtn = document.getElementById('uploadAnotherBtn');

    const predictedDigitDisplay = document.getElementById('predictedDigitDisplay');
    const confidenceText = document.getElementById('confidenceText');
    const confidenceBadge = document.getElementById('confidenceBadge');
    const resultOriginalImg = document.getElementById('resultOriginalImg');
    const resultTensorImg = document.getElementById('resultTensorImg');
    const probabilityBars = document.getElementById('probabilityBars');
    const sampleButtons = document.getElementById('sampleButtons');
    const toastContainer = document.getElementById('toastContainer');

    // --- Canvas Setup ---
    const ctx = drawingCanvas.getContext('2d');
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;

    function initCanvas() {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, drawingCanvas.width, drawingCanvas.height);
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = parseInt(brushSizeInput.value, 10);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
    }
    initCanvas();

    // Brush slider
    brushSizeInput.addEventListener('input', (e) => {
        const val = e.target.value;
        brushSizeVal.textContent = `${val}px`;
        ctx.lineWidth = parseInt(val, 10);
    });

    // Clear Canvas
    clearCanvasBtn.addEventListener('click', () => {
        initCanvas();
        hasDrawn = false;
        checkAnalyzeReady();
    });

    // Canvas drawing coordinates helper
    function getCanvasCoords(evt) {
        const rect = drawingCanvas.getBoundingClientRect();
        const scaleX = drawingCanvas.width / rect.width;
        const scaleY = drawingCanvas.height / rect.height;

        let clientX, clientY;
        if (evt.touches && evt.touches.length > 0) {
            clientX = evt.touches[0].clientX;
            clientY = evt.touches[0].clientY;
        } else {
            clientX = evt.clientX;
            clientY = evt.clientY;
        }

        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    function startDrawing(e) {
        e.preventDefault();
        isDrawing = true;
        const coords = getCanvasCoords(e);
        lastX = coords.x;
        lastY = coords.y;

        // Draw point on initial touch
        ctx.beginPath();
        ctx.arc(lastX, lastY, ctx.lineWidth / 2, 0, Math.PI * 2);
        ctx.fillStyle = '#FFFFFF';
        ctx.fill();

        hasDrawn = true;
        checkAnalyzeReady();
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const coords = getCanvasCoords(e);

        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(coords.x, coords.y);
        ctx.stroke();

        lastX = coords.x;
        lastY = coords.y;
    }

    function stopDrawing() {
        isDrawing = false;
    }

    drawingCanvas.addEventListener('mousedown', startDrawing);
    drawingCanvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDrawing);

    drawingCanvas.addEventListener('touchstart', startDrawing, { passive: false });
    drawingCanvas.addEventListener('touchmove', draw, { passive: false });
    window.addEventListener('touchend', stopDrawing);

    // --- Tabs Switch ---
    tabUploadBtn.addEventListener('click', () => {
        if (currentInputMode === 'upload') return;
        currentInputMode = 'upload';
        tabUploadBtn.classList.add('active');
        tabUploadBtn.setAttribute('aria-selected', 'true');
        tabDrawBtn.classList.remove('active');
        tabDrawBtn.setAttribute('aria-selected', 'false');

        uploadTabPane.classList.remove('hidden');
        drawTabPane.classList.add('hidden');
        checkAnalyzeReady();
    });

    tabDrawBtn.addEventListener('click', () => {
        if (currentInputMode === 'draw') return;
        currentInputMode = 'draw';
        tabDrawBtn.classList.add('active');
        tabDrawBtn.setAttribute('aria-selected', 'true');
        tabUploadBtn.classList.remove('active');
        tabUploadBtn.setAttribute('aria-selected', 'false');

        drawTabPane.classList.remove('hidden');
        uploadTabPane.classList.add('hidden');
        checkAnalyzeReady();
    });

    // --- Options Accordion ---
    toggleOptionsBtn.addEventListener('click', () => {
        const expanded = toggleOptionsBtn.getAttribute('aria-expanded') === 'true';
        toggleOptionsBtn.setAttribute('aria-expanded', String(!expanded));
        optionsContent.classList.toggle('hidden');
    });

    // --- File Upload & Drag-and-Drop ---
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropzone.addEventListener('click', () => {
        if (!selectedFile) {
            fileInput.click();
        }
    });

    dropzone.addEventListener('keydown', (e) => {
        if ((e.key === 'Enter' || e.key === ' ') && !selectedFile) {
            e.preventDefault();
            fileInput.click();
        }
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files && files.length > 0) {
            handleFileSelection(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });

    removeFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearUploadedFile();
    });

    function formatBytes(bytes, decimals = 1) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    function handleFileSelection(file) {
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp'];
        if (!validTypes.includes(file.type.toLowerCase()) && !/\.(png|jpe?g|webp)$/i.test(file.name)) {
            showToast('Unsupported file type. Please upload a PNG, JPG, or WEBP image.', 'error');
            return;
        }

        const maxSize = 5 * 1024 * 1024; // 5 MB
        if (file.size > maxSize) {
            showToast('File size exceeds the 5MB limit. Please choose a smaller image.', 'error');
            return;
        }

        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = formatBytes(file.size);

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            dropzoneEmpty.classList.add('hidden');
            dropzonePreview.classList.remove('hidden');
            checkAnalyzeReady();
        };
        reader.onerror = () => {
            showToast('Failed to read image file.', 'error');
            clearUploadedFile();
        };
        reader.readAsDataURL(file);
    }

    function clearUploadedFile() {
        selectedFile = null;
        fileInput.value = '';
        previewImage.src = '';
        dropzonePreview.classList.add('hidden');
        dropzoneEmpty.classList.remove('hidden');
        checkAnalyzeReady();
    }

    // --- Sample Buttons ---
    sampleButtons.addEventListener('click', (e) => {
        const btn = e.target.closest('.btn-sample');
        if (!btn) return;
        const digit = btn.getAttribute('data-sample');
        drawSampleDigit(digit);
    });

    function drawSampleDigit(digit) {
        // Switch to draw mode and draw the digit stroke cleanly
        if (currentInputMode !== 'draw') {
            tabDrawBtn.click();
        }
        initCanvas();
        ctx.fillStyle = '#FFFFFF';
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 24;

        const w = drawingCanvas.width;
        const h = drawingCanvas.height;
        const cx = w / 2;
        const cy = h / 2;

        ctx.beginPath();
        switch (digit) {
            case '0':
                ctx.ellipse(cx, cy, 60, 90, 0, 0, Math.PI * 2);
                ctx.stroke();
                break;
            case '2':
                ctx.arc(cx, cy - 45, 45, Math.PI * 1.1, 0);
                ctx.quadraticCurveTo(cx + 40, cy + 20, cx - 45, cy + 85);
                ctx.lineTo(cx + 55, cy + 85);
                ctx.stroke();
                break;
            case '3':
                ctx.arc(cx, cy - 40, 45, Math.PI * 1.2, Math.PI * 0.4);
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(cx, cy + 40, 50, -Math.PI * 0.4, Math.PI * 0.85);
                ctx.stroke();
                break;
            case '5':
                ctx.moveTo(cx + 45, cy - 80);
                ctx.lineTo(cx - 35, cy - 80);
                ctx.lineTo(cx - 40, cy - 10);
                ctx.arc(cx, cy + 20, 55, -Math.PI * 0.6, Math.PI * 0.7);
                ctx.stroke();
                break;
            case '7':
                ctx.moveTo(cx - 50, cy - 80);
                ctx.lineTo(cx + 50, cy - 80);
                ctx.lineTo(cx - 20, cy + 85);
                ctx.stroke();
                break;
            case '8':
                ctx.arc(cx, cy - 45, 40, 0, Math.PI * 2);
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(cx, cy + 45, 48, 0, Math.PI * 2);
                ctx.stroke();
                break;
            default:
                ctx.arc(cx, cy, 50, 0, Math.PI * 2);
                ctx.stroke();
        }

        hasDrawn = true;
        checkAnalyzeReady();
        showToast(`Sample digit ${digit} loaded onto the canvas!`, 'success');
    }

    // --- State Check ---
    function checkAnalyzeReady() {
        let isReady = false;
        if (currentInputMode === 'upload') {
            isReady = selectedFile !== null;
        } else {
            isReady = hasDrawn;
        }

        analyzeBtn.disabled = !isReady || isAnalyzing;
    }

    // --- Toast Notifications ---
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;

        const iconSvg = type === 'error'
            ? `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>`
            : `<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>`;

        toast.innerHTML = `${iconSvg}<span>${message}</span>`;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(10px)';
            setTimeout(() => toast.remove(), 250);
        }, 4000);
    }

    // --- Backend Health Check ---
    async function checkHealth() {
        try {
            const res = await fetch('/health');
            const data = await res.json();
            if (data.status === 'ok' && data.model_loaded) {
                healthPill.classList.remove('error');
                healthPill.classList.add('ready');
                healthStatusText.textContent = `Model Ready (${data.model_filename || 'ANN'})`;
            } else {
                healthPill.classList.remove('ready');
                healthPill.classList.add('error');
                healthStatusText.textContent = data.message || 'Model offline';
            }
        } catch (err) {
            healthPill.classList.remove('ready');
            healthPill.classList.add('error');
            healthStatusText.textContent = 'Server unreachable';
        }
    }
    checkHealth();

    // --- Inference Request ---
    analyzeBtn.addEventListener('click', async () => {
        if (isAnalyzing) return;

        const formData = new FormData();
        let displayDataUrl = '';

        if (currentInputMode === 'upload') {
            if (!selectedFile) return;
            formData.append('image', selectedFile);
            displayDataUrl = previewImage.src;
        } else {
            if (!hasDrawn) {
                showToast('Please draw a digit before analyzing.', 'error');
                return;
            }
            displayDataUrl = drawingCanvas.toDataURL('image/png');
            // Convert canvas to blob
            const blob = await new Promise(resolve => drawingCanvas.toBlob(resolve, 'image/png'));
            formData.append('image', blob, 'drawing.png');
        }

        formData.append('inversion_mode', inversionModeSelect.value);
        formData.append('center_digit', centerDigitCheck.checked ? 'true' : 'false');

        // UI Loading State
        isAnalyzing = true;
        analyzeBtn.disabled = true;
        analyzeSpinner.classList.remove('hidden');
        analyzeIcon.classList.add('hidden');
        analyzeBtnText.textContent = 'Analyzing...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok || !result.success) {
                const errorMsg = result.error || result.message || 'Prediction failed. Please try again.';
                showToast(errorMsg, 'error');
                return;
            }

            renderResults(result, displayDataUrl);
            showToast(`Predicted digit: ${result.predicted_digit} with ${result.confidence}% confidence`, 'success');

        } catch (err) {
            console.error('Inference error:', err);
            showToast('Network error while contacting prediction service.', 'error');
        } finally {
            isAnalyzing = false;
            analyzeSpinner.classList.add('hidden');
            analyzeIcon.classList.remove('hidden');
            analyzeBtnText.textContent = 'Analyze Digit';
            checkAnalyzeReady();
        }
    });

    // --- Render Results ---
    function renderResults(data, originalDataUrl) {
        // Toggle view
        resultsEmpty.classList.add('hidden');
        resultsBody.classList.remove('hidden');

        // Predicted Digit
        predictedDigitDisplay.textContent = data.predicted_digit;

        // Confidence
        const confidence = parseFloat(data.confidence).toFixed(1);
        confidenceText.textContent = `${confidence}% Confidence`;

        // Images preview
        resultOriginalImg.src = originalDataUrl;
        if (data.preprocessed_image) {
            resultTensorImg.src = data.preprocessed_image;
        }

        // Probability Chart for digits 0-9
        probabilityBars.innerHTML = '';
        const probs = data.probabilities || {};

        for (let digit = 0; digit <= 9; digit++) {
            const rawProb = probs[digit] !== undefined ? probs[digit] : 0;
            const probPct = parseFloat(rawProb).toFixed(1);
            const isTop = digit === data.predicted_digit;

            const row = document.createElement('div');
            row.className = `prob-row ${isTop ? 'top' : ''}`;
            row.innerHTML = `
                <span class="prob-digit">${digit}</span>
                <div class="prob-track">
                    <div class="prob-fill" style="width: 0%" data-width="${probPct}%"></div>
                </div>
                <span class="prob-val">${probPct}%</span>
            `;
            probabilityBars.appendChild(row);
        }

        // Trigger smooth width animations
        requestAnimationFrame(() => {
            probabilityBars.querySelectorAll('.prob-fill').forEach(fill => {
                fill.style.width = fill.getAttribute('data-width');
            });
        });

        // Scroll to results if mobile
        if (window.innerWidth < 960) {
            resultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    // --- Clear & Reset ---
    function clearPrediction() {
        resultsBody.classList.add('hidden');
        resultsEmpty.classList.remove('hidden');
        predictedDigitDisplay.textContent = '-';
        confidenceText.textContent = '0.0% Confidence';
        probabilityBars.innerHTML = '';
        resultOriginalImg.src = '';
        resultTensorImg.src = '';
    }

    clearPredictionBtn.addEventListener('click', () => {
        clearPrediction();
        showToast('Prediction results cleared.', 'info');
    });

    uploadAnotherBtn.addEventListener('click', () => {
        clearPrediction();
        clearUploadedFile();
        initCanvas();
        hasDrawn = false;
        if (currentInputMode === 'upload') {
            fileInput.click();
        }
    });
});
