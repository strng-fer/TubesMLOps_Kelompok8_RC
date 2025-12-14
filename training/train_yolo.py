from ultralytics import YOLO
import wandb
import shutil
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup W&B authentication
wandb_api_key = os.getenv('WANDB_API_KEY')
if wandb_api_key:
    wandb.login(key=wandb_api_key)
    # Initialize W&B
    wandb.init(project="pothole-detection", name="yolo-model-fix")
    use_wandb = True
else:
    print("WANDB_API_KEY not found. Running without W&B tracking.")
    use_wandb = False

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data="../data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    project="runs/pothole",
    name="exp1.0"
)

# Log training metrics to W&B if enabled
if use_wandb:
    # Access training results metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        wandb.log({
            "train/box_loss": metrics.get("train/box_loss", 0),
            "train/cls_loss": metrics.get("train/cls_loss", 0),
            "train/dfl_loss": metrics.get("train/dfl_loss", 0),
            "metrics/precision": metrics.get("metrics/precision(B)", 0),
            "metrics/recall": metrics.get("metrics/recall(B)", 0),
            "metrics/mAP50": metrics.get("metrics/mAP50(B)", 0),
            "metrics/mAP50-95": metrics.get("metrics/mAP50-95(B)", 0),
        })
    else:
        # Fallback: log basic info
        wandb.log({"training_completed": True, "epochs": 50})

# Predict on test set
test_results = model.predict(
    source="../dataset/images/test",
    conf=0.25,
    save=True
)

# Log test predictions metrics to W&B if enabled
if use_wandb:
    # Calculate and log test metrics if available
    if test_results:
        # Example: log number of detections
        total_detections = sum(len(result.boxes) for result in test_results)
        wandb.log({"test/total_detections": total_detections})
        print(f"Logged {total_detections} detections on test set to W&B")

# Save the final model as best.pt directly
model.save('../models/best.pt')

# No need for separate copy since we save directly to best.pt

# Log model to W&B Registry if W&B is enabled
if use_wandb:
    # Log the best model to W&B
    wandb.log_model(
        path="../models/best.pt",
        name="pothole-yolov8n",
        aliases=["latest", "v1"]
    )
    print("Model logged to W&B Registry.")

# Finish W&B if used
if use_wandb:
    wandb.finish()