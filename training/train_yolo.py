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
    wandb.init(project="pothole-detection", name="yolo-model-v1")
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
    name="exp1"
)

# Predict on test set
model.predict(
    source="../dataset/images/test",
    conf=0.25,
    save=True
)

# Save the final model
model.save('../models/pothole_yolov8n.pt')

# Copy the best model to models/ (overwrite best.pt)
shutil.copy("runs/pothole/exp1/weights/best.pt", "../models/best.pt")

# Finish W&B if used
if use_wandb:
    wandb.finish()