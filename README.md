# ADAS & AI/ML Prototype

This project is a prototype for an Advanced Driver Assistance System (ADAS) that demonstrates sensor fusion, object detection, and collision avoidance using multiple sensors (Camera, Radar/LIDAR) and AI/ML techniques.

## Project Overview

In this project, we:
- **Preprocess Data:**
  - Load and process a small subset of the KITTI dataset (2011_09_26_drive_0048) including images, calibration data, and LIDAR point clouds.
  - Use calibration files to project LIDAR data into the camera image plane.
  
- **Object Detection:**  
  - Use a pre-trained YOLOv5 model (via PyTorch Hub) to detect objects (cars, pedestrians, etc.) in the camera images.
  
- **Sensor Fusion:**  
  - Combine camera detections with simulated radar/LIDAR data to produce a unified output.
  
- **Decision Making:**  
  - Implement simple collision avoidance logic by comparing object distances with a threshold, triggering a "Brake" command when necessary.

## Installation

1. **Clone the repository** (or download and extract the project folder).

pip install -r requirements.txt
# ADAS-AI-ML-Prototype
