# 🚧 MLOps End-to-End Blueprint: Pothole Detection System

Proyek ini merupakan implementasi **MLOps end-to-end** untuk sistem **Deteksi Jalan Berlubang (Pothole Detection)** menggunakan *Deep Learning Object Detection* dan di-deploy sebagai **Web API berbasis FastAPI**. Proyek ini dirancang sebagai blueprint pembelajaran dan praktik MLOps yang selaras dengan workflow industri.

---

## 👤 Informasi Proyek

* **Program Studi**: Sains Data – ITERA
* **Tahun**: 2025
* **Topik**: Computer Vision & MLOps
* **Framework Utama**: YOLO (Ultralytics)
* **Experiment Tracking**: Weights & Biases (W&B)

---

## 🎯 Tujuan Proyek

1. Memahami workflow **MLOps end-to-end**
2. Mengimplementasikan pipeline **training–tracking–deployment–monitoring**
3. Membangun sistem deteksi jalan berlubang yang siap digunakan secara online

---

## 🧠 Problem Statement

Kerusakan jalan berupa **lubang (pothole)** menjadi salah satu penyebab utama kecelakaan dan penurunan kualitas infrastruktur. Deteksi manual memakan waktu dan biaya tinggi. Oleh karena itu, dibutuhkan sistem otomatis berbasis **Computer Vision** yang dapat mendeteksi jalan berlubang dari citra.

---

## 💡 Solusi

Mengembangkan sistem deteksi jalan berlubang berbasis **Object Detection (YOLO)** dengan pendekatan **MLOps**, sehingga model dapat:

* Dilatih dan dievaluasi secara terukur
* Dicatat eksperimennya (experiment tracking)
* Dideploy sebagai layanan API
* Siap untuk monitoring dan pengembangan lanjutan

---

## 🗂️ Struktur Project

```
project-root/
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── data.yaml
├── models/
│   └── best.pt
│
├── training/
│   └── train_yolo.py
│
├── inference/
│   └── predict.py
│
├── templates/
│   └── index.html          # UI template
│
├── saved_images/           # Auto-saved detection images
│
├── app.py                  # FastAPI deployment with UI
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔄 Workflow MLOps

### 1. Data Ingestion

* Dataset citra jalan berlubang
* Anotasi bounding box menggunakan format **YOLO**

### 2. Data Preparation & Preprocessing

* Resize & normalisasi image
* Data split: train / validation / test

### 3. Modeling

* Model: **YOLOv8 / YOLOv11**
* Task: Object Detection

### 4. Training & Experiment Tracking

* Training menggunakan Ultralytics
* Logging metrik ke **Weights & Biases**:

  * Training & validation loss
  * Precision, Recall, mAP
  * Sample prediction

### 5. Model Evaluation

* Evaluasi pada validation & test set
* Analisis hasil prediksi dan error

### 6. Model Versioning

* Model terbaik disimpan sebagai `best.pt`
* Siap untuk deployment

### 7. Deployment (Online Inference)

* Model di-deploy menggunakan **FastAPI**
* Endpoint `/predict` untuk inference gambar

### 8. Monitoring

* **Health Check**: `/health` endpoint for model status
* **Metrics Collection**: `/metrics` endpoint for response time and error rates
* **Performance Drift Detection**: `/drift_check` endpoint evaluates model on test set
* **Retraining Triggers**: `/retrain` endpoint for automated model updates

---

## 📊 Metrics Evaluasi

* **Precision**
* **Recall**
* **mAP@0.5**
* **mAP@0.5:0.95**

---

## 🧪 Experiment Tracking (W&B)

Semua eksperimen dicatat menggunakan **Weights & Biases**, meliputi:

* Konfigurasi training
* Kurva loss
* Metrik evaluasi
* Sample hasil prediksi

🔗 Dashboard W&B: *(link sesuai akun masing-masing)*

---

## 🚀 Deployment dengan FastAPI

### Menjalankan API secara lokal

```bash
uvicorn app:app --reload
```

### User Interface (UI)

Sistem ini menyediakan **antarmuka pengguna web** untuk pengguna akhir yang dapat:

* **Memilih versi model**: Pilih model YOLO yang berbeda (jika tersedia)
* **Mengatur confidence threshold**: Atur tingkat kepercayaan minimum untuk deteksi
* **Tampilan video live**: Streaming video real-time dari webcam dengan deteksi otomatis
* **Penyimpanan gambar otomatis**: Setiap objek terdeteksi akan disimpan sebagai gambar dengan delay 5 detik untuk menghindari duplikasi

#### Cara Menggunakan UI:

1. Jalankan aplikasi: `python app.py`
2. Buka browser ke `http://localhost:8000`
3. Pilih pengaturan model dan confidence
4. Lihat video live dengan deteksi real-time
5. Gambar deteksi tersimpan di folder `saved_images/`

### Endpoint Tambahan untuk UI

* **GET** `/` - Halaman utama UI
* **POST** `/update_settings` - Update pengaturan model dan confidence
* **GET** `/video_feed` - Streaming video dengan deteksi real-time

---

## 🐳 Docker (Opsional)

```bash
docker build -t pothole-api .
docker run -p 8000:8000 pothole-api
```

---

## 🚂 Railway Deployment (Direkomendasikan)

Railway adalah platform ideal untuk deploy aplikasi Python + ML dengan persistent processes dan support untuk long-running tasks.

### 📋 Persiapan:

1. **Railway Account**: 
   - Daftar di [railway.app](https://railway.app)
   - Connect GitHub account

2. **Repository**: Pastikan kode di-push ke GitHub

### 🚀 Deploy via GitHub Integration:

1. **Connect Repository**:
   - Login ke Railway dashboard
   - Click "New Project" → "Deploy from GitHub repo"
   - Pilih repository Anda

2. **Auto-Deployment**:
   - Railway akan auto-detect Python app dari `requirements.txt` atau `Dockerfile`
   - Build dan deploy otomatis

3. **Set Environment Variables**:
   - Pergi ke project settings → Variables
   - Add: `WANDB_API_KEY` = "your_api_key_here"
   - Add: `PORT` = "8000" (opsional, Railway auto-set)

### 🐳 Deploy via Docker (Advanced):

Railway support Docker deployment menggunakan `Dockerfile` yang sudah ada.

### 🔧 Railway CLI (Alternatif):

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Set environment variables
railway variables set WANDB_API_KEY=your_key_here

# Deploy
railway up
```

### 📊 Monitoring & Logs:

- **Logs**: View real-time logs di Railway dashboard
- **Metrics**: CPU, Memory, Network usage
- **Scaling**: Auto-scaling berdasarkan load

### 🌐 Custom Domain:

- Di project settings → Domains
- Add custom domain jika perlu

### 📁 File Konfigurasi yang Sudah Disiapkan:

- **`Dockerfile`**: Container configuration
- **`railway.json`**: Railway deployment config
- **`.env.example`**: Template environment variables
- **`.gitignore`**: Git ignore rules

### 💰 Upgrade Plan Railway:

Jika image masih >4GB, upgrade ke **Hobby Plan** ($5/bulan):
- Image size limit: **8GB**
- RAM: 8GB
- Disk: 5GB

**Cara Upgrade:**
1. Pergi ke Railway dashboard
2. Project Settings → Billing
3. Upgrade ke Hobby Plan

### 🔄 Redeploy Setelah Optimasi:

```bash
git add .
git commit -m "Optimize Docker for smaller image size"
git push origin main
```

Railway akan auto-rebuild dengan Dockerfile yang baru.

---

## 📌 Tools & Technology

| Komponen   | Tools              |
| ---------- | ------------------ |
| Data & CV  | OpenCV, PIL        |
| Model      | YOLO (Ultralytics) |
| Tracking   | Weights & Biases   |
| Backend    | FastAPI            |
| Deployment | Docker             |
| Versioning | GitHub             |

---

## 📈 Future Improvement

* CI/CD dengan GitHub Actions
* Model registry (MLflow / W&B Artifacts)
* Monitoring inference (latency, drift)
* Frontend dashboard

---

## 📅 Deadline Proyek

**15 Desember 2025**

---

## 🏁 Kesimpulan

Proyek ini menunjukkan bagaimana model Machine Learning tidak hanya dilatih, tetapi juga **dikelola, dilacak, dan dideploy** secara sistematis menggunakan pendekatan **MLOps**.

Blueprint ini dapat digunakan sebagai dasar untuk proyek MLOps lainnya, khususnya di domain **Computer Vision**.

---

✨ *End of README Blueprint*
