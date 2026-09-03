---
title: Dental AI Segmentation Viewer
emoji: 🦷
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
short_description: Deep learning dental X-ray instance segmentation.
---

# 🦷 Dental AI Segmentation Viewer

A web application that analyzes panoramic dental X-ray images using dual PyTorch YOLO instance segmentation models (`anatomy_model.pt` and `dental_model.pt`) and visualizes anatomical structures and dental findings with anti-collision callout badges and interactive filters.

## 🚀 Features

- **Dual-Model Inference**: Runs anatomical structure segmentation and dental findings segmentation in parallel.
- **Anti-Collision Callouts**: Leader lines and numbered callouts automatically adjust positions to prevent badge overlap.
- **Multi-Column Dynamic Legend**: Legend entries scale across columns without squishing text.
- **Interactive Sidebar Cards**: Real-time hover synchronization between sidebar cards and canvas elements.

## 🛠️ Stack & Architecture

- **Backend**: Python 3.10, Flask, Gunicorn, OpenCV, NumPy, Pillow
- **Machine Learning**: Ultralytics YOLO segmentation models (`.pt`)
- **Frontend**: HTML5 Canvas, Vanilla JS, Modern CSS3
- **Containerization**: Docker (deployed on Hugging Face Spaces)
