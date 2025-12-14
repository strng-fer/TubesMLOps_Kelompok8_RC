from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
import json
import os
from datetime import datetime
from ultralytics import YOLO
import numpy as np

app = FastAPI()

# Templates
templates = Jinja2Templates(directory="templates")

# Global variables for settings
current_model = "pothole_yolov8n.pt"
confidence_threshold = 0.5
model = None
last_frame = None  # Store the latest frame for feedback

def load_model():
    global model
    try:
        model_path = f"models/{current_model}"
        if os.path.exists(model_path):
            model = YOLO(model_path)
            print(f"Model {current_model} loaded successfully")
        else:
            print(f"Model {model_path} not found")
            model = None
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Load initial model
load_model()

def generate_frames():
    try:
        cap = cv2.VideoCapture(0)  # Use webcam
        if not cap.isOpened():
            print("Cannot open webcam")
            # Return a placeholder frame
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Webcam not available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            return
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Store the latest frame globally for feedback
            global last_frame
            last_frame = frame.copy()
            
            # Perform inference if model is loaded
            if model:
                try:
                    results = model(frame, conf=confidence_threshold)
                    
                    # Draw detections
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = box.conf[0].item()
                            
                            # Draw rectangle
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            
                            # Draw confidence
                            cv2.putText(frame, f"{conf:.2f}", (int(x1), int(y1)-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Inference error: {e}")
                    cv2.putText(frame, "Inference error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        print(f"Video generation error: {e}")
    finally:
        try:
            cap.release()
        except:
            pass

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "current_model": current_model,
        "confidence_threshold": confidence_threshold
    })

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), 
                           media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/update_settings")
async def update_settings(
    model_version: str = Form(...),
    confidence: float = Form(...)
):
    global current_model, confidence_threshold
    current_model = model_version
    confidence_threshold = confidence
    
    # Reload model
    load_model()
    
    return {
        "model": current_model,
        "confidence": confidence_threshold
    }

@app.post("/feedback")
async def submit_feedback(has_pothole: bool = Form(...)):
    # Load existing feedback log
    feedback_file = "feedback_log.json"
    try:
        if os.path.exists(feedback_file) and os.path.getsize(feedback_file) > 0:
            with open(feedback_file, 'r') as f:
                feedback_log = json.load(f)
        else:
            feedback_log = []
    except json.JSONDecodeError:
        print("Invalid JSON in feedback_log.json, starting fresh")
        feedback_log = []
    
    # Add new feedback
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "has_pothole": has_pothole
    }
    feedback_log.append(feedback_entry)
    
    # Save feedback log
    with open(feedback_file, 'w') as f:
        json.dump(feedback_log, f, indent=2)
    
    # Save current frame as image if pothole detected
    saved_image = False
    if has_pothole:
        try:
            global last_frame
            if last_frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"saved_images/pothole_{timestamp}.jpg"
                os.makedirs("saved_images", exist_ok=True)
                cv2.imwrite(image_path, last_frame)
                saved_image = True
                print(f"Saved image: {image_path}")
            else:
                print("No frame available for saving")
        except Exception as e:
            print(f"Error saving image: {e}")
    
    return {"saved_image": saved_image}
