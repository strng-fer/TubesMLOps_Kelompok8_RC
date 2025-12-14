FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (minimal for OpenCV and YOLO)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install (exclude heavy optional deps)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --only-binary=all

# Copy application code
COPY app.py .
COPY data.yaml .
COPY templates/ ./templates/
COPY training/train_yolo.py ./training/

# Create necessary directories
RUN mkdir -p models saved_images

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]