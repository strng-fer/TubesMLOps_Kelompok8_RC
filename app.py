from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import logging
import time
from collections import defaultdict
import subprocess
import threading
import os
from datetime import datetime

# Set up logging for monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pothole Detection API", description="MLOps Pothole Detection using YOLO")

# Mount static files and templates
# app.mount("/static", StaticFiles(directory="static"), name="static")  # Commented out since no static files needed
templates = Jinja2Templates(directory="templates")

# Create directories for saved images
os.makedirs("saved_images", exist_ok=True)

# Load YOLO model with error handling
try:
    model = YOLO("models/best.pt")
    model_loaded = True
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model_loaded = False
    model = None

# Global settings for UI
current_model = "best.pt"
confidence_threshold = 0.5
last_detection_time = 0
detection_delay = 5  # seconds

# Simple in-memory metrics
metrics = defaultdict(int)
response_times = []

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    metrics["total_requests"] += 1
    response_times.append(process_time)

    if response.status_code >= 400:
        metrics["errors"] += 1

    logger.info(f"Request: {request.method} {request.url} - Status: {response.status_code} - Time: {process_time:.4f}s")

    return response

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring model status."""
    if model_loaded:
        return {"status": "healthy", "model_loaded": True}
    else:
        raise HTTPException(status_code=503, detail="Model not loaded")

@app.get("/metrics")
async def get_metrics():
    """Basic metrics endpoint for monitoring."""
    avg_time = sum(response_times) / len(response_times) if response_times else 0
    return {
        "total_requests": metrics["total_requests"],
        "errors": metrics["errors"],
        "avg_response_time": avg_time,
        "model_loaded": model_loaded
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not available")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = np.array(image)

        # Perform detection
        results = model(image)

        # Extract bounding boxes and confidence scores
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class": cls
                })

        # Log prediction for monitoring
        logger.info(f"Prediction completed: {len(detections)} detections")

        return JSONResponse(content={"detections": detections})

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/drift_check")
async def drift_check():
    """Check model performance drift by evaluating on test set."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        # Run validation on test set
        results = model.val(data="data.yaml", split="test")

        # Extract metrics
        metrics_result = {
            "precision": results.results_dict.get("metrics/precision(B)", 0),
            "recall": results.results_dict.get("metrics/recall(B)", 0),
            "map50": results.results_dict.get("metrics/mAP50(B)", 0),
            "map50_95": results.results_dict.get("metrics/mAP50-95(B)", 0)
        }

        logger.info(f"Drift check completed: {metrics_result}")

        return JSONResponse(content={"drift_metrics": metrics_result})

    except Exception as e:
        logger.error(f"Drift check error: {e}")
        raise HTTPException(status_code=500, detail="Drift check failed")

@app.post("/update_settings")
async def update_settings(model_version: str = Form(...), confidence: float = Form(...)):
    """Update model and confidence settings."""
    global current_model, confidence_threshold, model, model_loaded
    
    confidence_threshold = confidence
    
    if model_version != current_model:
        current_model = model_version
        try:
            model = YOLO(f"models/{model_version}")
            model_loaded = True
            logger.info(f"Model switched to {model_version}")
        except Exception as e:
            logger.error(f"Failed to load model {model_version}: {e}")
            model_loaded = False
    
    return {"message": "Settings updated", "model": current_model, "confidence": confidence_threshold}

def generate_frames():
    """Generate video frames with real-time detection."""
    global last_detection_time
    
    cap = cv2.VideoCapture(0)  # Use webcam
    
    if not cap.isOpened():
        # If webcam not available, return error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Webcam not available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame_bytes = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Perform detection if model is loaded
        if model_loaded and model:
            results = model(frame, conf=confidence_threshold)
            
            # Draw detections
            annotated_frame = results[0].plot()
            
            # Check for detections and save image if needed
            detections = len(results[0].boxes)
            if detections > 0:
                current_time = time.time()
                if current_time - last_detection_time > detection_delay:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"saved_images/detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    last_detection_time = current_time
                    logger.info(f"Saved detection image: {filename}")
        else:
            annotated_frame = frame
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
async def video_feed():
    """Video streaming endpoint."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main UI page for pothole detection."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "current_model": current_model,
        "confidence_threshold": confidence_threshold
    })

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)