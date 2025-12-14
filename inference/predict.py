from ultralytics import YOLO
import cv2

def predict_image(image_path, model_path="../models/pothole_yolov8n.pt"):
    # Load model
    model = YOLO(model_path)

    # Load image
    image = cv2.imread(image_path)

    # Perform inference
    results = model(image)

    # Process results
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

    return detections

if __name__ == "__main__":
    # Example usage
    detections = predict_image("path/to/image.jpg")
    print(detections)