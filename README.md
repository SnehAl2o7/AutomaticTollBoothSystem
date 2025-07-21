# 🚗 Automatic Toll Booth System 💰

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/ObjectDetection-YOLOv8-purple)
![Flask](https://img.shields.io/badge/Backend-Flask-orange)
![OCR](https://img.shields.io/badge/OCR-EasyOCR-red)

> An AI-powered smart toll booth system that detects vehicles, classifies them, charges the correct toll fee, and automates gate control – all without manual intervention. Fast, accurate, and future-ready! 🌉💡

---

## 🚀 Project Highlights

| 🔧 Feature                   | ⚙️ Description |
|-----------------------------|----------------|
| 🚘 **Vehicle Detection**    | Detects incoming vehicles using **YOLOv8** object detection |
| 🧠 **Vehicle Classification**| Classifies vehicle type (car, truck, bike) using model inference |
| 💵 **Toll Calculation**     | Dynamically charges toll based on vehicle type |
| 🧾 **OCR for Plate Reading**| Uses **EasyOCR** to read number plates (optional extension) |
| 🛑 **Barrier Control**      | Opens the gate automatically if toll is paid |
| 🗃 **Logging**              | Saves vehicle entries with timestamps |
| 🌐 **Web Interface**        | Built with **Flask + HTML/CSS** for user interaction and visualization |

---

## 🧠 Tech Stack

| Category              | Technologies Used |
|-----------------------|-------------------|
| **🔍 Object Detection**  | [YOLOv8](https://github.com/ultralytics/ultralytics) |
| **🔤 OCR**              | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| **🌐 Backend**         | Flask, Python 3.8+ |
| **📦 Libraries**       | OpenCV, NumPy, Pillow, Ultralytics, EasyOCR, Flask |
| **🗄️ Database**        | SQLite (can upgrade to MongoDB/PostgreSQL) |
| **🖥 UI**              | HTML, CSS, Bootstrap (optional React for dashboard) |

---

🧰 Installation & Setup <br>
1️⃣ Clone the Repository <br>
Edit
git clone https://github.com/yourusername/AutomaticTollBoothSystem.git <br>
cd AutomaticTollBoothSystem

## 📊 System Flow

```mermaid
graph TD
    A[Vehicle Arrives at Toll Booth] --> B[Camera Captures Image]
    B --> C[YOLOv8 Detects Vehicle]
    C --> D[Classify Vehicle Type]
    D --> E[Calculate Toll Fee]
    E --> F{Payment Success?}
    F -- Yes --> G[Log Entry and Open Gate]
    F -- No --> H[Display Payment Required Message]
    G --> I[Allow Vehicle to Pass]
    H --> I
