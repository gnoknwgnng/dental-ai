---
title: Dental AI Segmentation Viewer
emoji: 🦷
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: gradio_app.py
short_description: Deep learning dental X-ray instance segmentation.
---

# 🦷 Dental AI Segmentation Viewer

A web application that analyzes panoramic dental X-ray images using dual PyTorch YOLO instance segmentation models (`anatomy_model.pt` and `dental_model.pt`) and visualizes anatomical structures and dental findings.

## 🚀 Features

- **Dual-Model Inference**: Runs anatomical structure segmentation and dental findings segmentation in parallel.
- **Visual Contours**: Polygons and labels rendered over detected anatomical landmarks and pathologies.
- **Interactive Interface**: Powered by Gradio and PyTorch.
