# 👁️ Smart Vision AI

## 🌟[Live Demo](https://smart-vision-ai-uaay8nva3t9jk6een9eczx.streamlit.app/)

- [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
- [![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
- [![YOLOv8](https://img.shields.io/badge/YOLOv8-Powered-orange.svg)](https://github.com/ultralytics/ultralytics)

> 🚀 **Real-time Object Detection Dashboard powered by YOLOv8**

![Demo](https://img.shields.io/badge/Demo-Live-brightgreen)
![80 Classes](https://img.shields.io/badge/COCO-80%20Classes-blue)
![Cloud Ready](https://img.shields.io/badge/Cloud-Ready-success)

---

## 🌟 Overview

**Smart Vision AI** is a production-ready computer vision application that detects 80 different object classes in real-time using state-of-the-art YOLOv8 architecture. It features an interactive analytics dashboard, multi-format input support, and cloud-optimized performance.

## ✨ Key Features

- 🔍 **Real-time Object Detection** — Powered by YOLOv8 with 80 COCO classes
- 📊 **Interactive Analytics Dashboard** — Beautiful Plotly charts and metrics
- 🖼️ **Multi-format Support** — Process images (PNG, JPG) and videos (MP4, AVI, MOV)
- 🎯 **Dynamic Confidence Tuning** — Adjust detection threshold in real-time
- 🤖 **Multiple Model Variants** — Choose between Nano (fast), Small (balanced), or Medium (accurate)
- ☁️ **Cloud-Optimized** — Runs smoothly on Streamlit Cloud's free tier
- 📈 **Detailed Reporting** — Export detection data as CSV for further analysis

## 🛠️ Tech Stack

|        Component     |     Technology       |
|----------------------|----------------------|
| **Object Detection** | YOLOv8 (Ultralytics) |
| **Web Interface**    | Streamlit            |
| **Image Processing** | OpenCV, Pillow       |
| **Analytics**        | Plotly, Pandas       |
| **Deployment**       | Streamlit Cloud      |

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/javascriptsali/smart-vision-ai.git
cd smart-vision-ai
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/Scripts/activate  # Windows Git Bash
# or: source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app/streamlit_app.py
```

- The app will open automatically at [http://localhost:8501]

## 📁 Project Structure

```tree
smart-vision-ai/
├── app/
│   └── streamlit_app.py      # Main Streamlit UI
├── src/
│   ├── detector.py            # YOLO detection engine
│   └── analytics.py           # Charts & reporting
├── data/
│   ├── uploads/               # Temporary uploads
│   └── outputs/               # Processed results
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── requirements.txt
├── .gitignore
└── README.md
```

### 🎯 Use Cases

- 🏪 Retail Analytics — Customer counting & behavior analysis
- 🚦 Traffic Monitoring — Vehicle detection & counting
- 🔒 Security Systems — Intrusion detection & surveillance
- 🏭 Quality Control — Defect detection in manufacturing
- 🐾 Wildlife Monitoring — Animal detection & tracking

## ⚙️ Configuration

### Upload Size Limit

Edit `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 100  # in MB
```

### Model Selection

| Model | Size | Speed  | Accuracy |
|-------|------|--------|----------|
|YOLOv8n| 6 MB |⚡⚡⚡  | ***      |  
|YOLOv8s| 22 MB|⚡⚡    |  ****    |
|YOLOv8m| 52 MB|  ⚡    | *****    |

## ⚠️ Cloud Limitations

- When deployed on Streamlit Cloud's free tier:
- Use short videos (under 30 seconds)
- Use low-resolution images
- Large files may cause memory issues

## 📝 License

- This project is licensed under the MIT License — see the LICENSE file for details.

## 🤝 Contributing

- Contributions, issues, and feature requests are welcome!

## 📧 Contact

- [Github Profile](https://github.com/javascriptsali)

- [Project Link](https://github.com/javascriptsali/smart-vision-ai)

## ⭐ **If you find this project useful, consider giving it a star!**
