# Hand Detection + Finger Counting (MediaPipe + OpenCV)

A small Python app that detects your hand via webcam, draws **landmarks**, and counts **raised fingers** (including a robust thumb heuristic).  
Tested on **Windows 10/11**, **Python 3.12**, inside a virtual environment.

---

## ✨ Features
- Real-time detection of 1–2 hands
- Landmark and connection overlay
- **Reliable finger counting**, including the **thumb**
- Mirror view for natural UX
- FPS overlay
- Simple code structure (easy to add gestures/actions/screenshots)

---

## 🧰 Requirements
- Windows 10+ (x64)
- Python **3.12**
- A working webcam
- (Recommended) Microsoft **Visual C++ Redistributable** 2015–2022 (x64)

> **Important:** Do **not** install TensorFlow in this environment. MediaPipe may try to import TF internals and break due to `protobuf` mismatches.

---

## 🚀 Quick Setup (Command Prompt / CMD)

1) Create (or reuse) the venv:
```bat
py -3.12 -m venv "%USERPROFILE%\venvs\handai312"
