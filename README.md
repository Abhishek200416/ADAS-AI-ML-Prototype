

# ADAS & AI/ML Prototype

This repository demonstrates an **Advanced Driver Assistance System (ADAS)** prototype using the **KITTI dataset** for object detection, sensor fusion, and collision avoidance. Two different main scripts showcase distinct refinement approaches:
- **2D Contour-Based Refinement** (`main.py`)
- **3D LIDAR-Based Refinement** (`main1.py`)

Both approaches combine **YOLOv5 object detection** with sensor fusion techniques to generate collision avoidance decisions and produce annotated output images.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Prerequisites & Installation](#prerequisites--installation)  
3. [Data & Calibration Files](#data--calibration-files)  
4. [Key Features](#key-features)  
5. [Project Structure](#project-structure)  
6. [Workflow Steps](#workflow-steps)  
   - [Step 1: Data Preprocessing & Calibration](#step-1-data-preprocessing--calibration)  
   - [Step 2: Object Detection](#step-2-object-detection)  
   - [Step 3: Sensor Fusion & Refinement](#step-3-sensor-fusion--refinement)  
   - [Step 4: Collision Avoidance & Decision Making](#step-4-collision-avoidance--decision-making)  
   - [Step 5: Visualization & Output](#step-5-visualization--output)  
7. [How to Run the Scripts](#how-to-run-the-scripts)  
   - [2D Contour-Based Refinement (main.py)](#2d-contour-based-refinement-mainpy)  
   - [3D LIDAR-Based Refinement (main1.py)](#3d-lidar-based-refinement-main1py)  
8. [Sample Output](#sample-output)  
9. [References](#references)

---

## Project Overview

The prototype integrates camera images and LIDAR data from the KITTI dataset to demonstrate:

- **Data Preprocessing:**  
  Loading camera frames, LIDAR point clouds, and calibration files; filtering LIDAR points (removing ground points, clipping by distance); and synchronizing sensor data.

- **Object Detection:**  
  Using a pre-trained **YOLOv5** model (via PyTorch Hub) to detect objects such as cars and pedestrians. The model outputs bounding boxes, confidence scores, and class labels.

- **Sensor Fusion & Refinement:**  
  Two refinement approaches are implemented:
  - **2D Contour-Based:** Uses OpenCV contours to refine the bounding boxes purely in the image plane.
  - **3D LIDAR-Based:** Projects LIDAR points onto the image plane using calibration matrices and refines bounding boxes if sufficient LIDAR points are present within the initial YOLO boxes.

- **Collision Avoidance:**  
  A basic decision-making module compares object distances (or refined bounding box metrics) against a threshold to decide whether to issue a "Brake" or "Continue" command.

- **Visualization:**  
  The final annotated output image overlays the original YOLO detections (blue boxes), the refined detections (green boxes), color-coded LIDAR points (colors indicate distance), and decision text (e.g., “Decision: Continue”).

---

## Prerequisites & Installation

1. **Clone or Download** the repository.
2. **Install Dependencies:**  
   Run the following command to install the necessary packages:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` includes:
   - `torch` and `torchvision` for running YOLOv5.
   - `opencv-python` for image processing and visualization.
   - `numpy` for numerical computations.
   - `matplotlib` (optional) for additional plotting.

3. **KITTI Dataset Subset:**  
   Download and place the KITTI data subset (`2011_09_26_drive_0048_sync`) along with the calibration files (`2011_09_26_calib`) into the `data` folder as specified below.

---

## Data & Calibration Files

- **KITTI Data:**  
  - **Images:** Located in `data/2011_09_26_drive_0048_sync/` (e.g., `image_02/data/` contains PNG files).  
  - **LIDAR Data:** Located in `data/2011_09_26_drive_0048_sync/velodyne_points/data/` (each frame is a `.bin` file).

- **Calibration Files:**  
  - Located in `data/2011_09_26_calib/2011_09_26/`.  
  - Files include `calib_velo_to_cam.txt` and `calib_cam_to_cam.txt`, which are essential for projecting LIDAR points into the camera coordinate system.

---

## Key Features

- **YOLOv5 Object Detection:**  
  Detects objects with bounding boxes, class labels, and confidence scores.
  
- **2D Contour-Based Refinement:**  
  Uses OpenCV contour detection to refine the initial YOLO detections directly in the image plane.
  
- **3D LIDAR-Based Refinement:**  
  Projects LIDAR points onto the image using calibration matrices to refine bounding boxes based on 3D sensor data.
  
- **Collision Avoidance:**  
  Implements a simple logic to decide whether to "Brake" or "Continue" based on the sensor-fused data.
  
- **Visualization:**  
  Overlays detections, refined boxes, color-coded LIDAR points, and decision text on the final image output.

---

## Project Structure

```
ADAS_Project/
├── data/
│   ├── 2011_09_26_calib/          # Calibration files
│   └── 2011_09_26_drive_0048_sync/ # KITTI images and LIDAR .bin files
├── src/                           # Source code modules
│   ├── data_preprocessing.py      # Loads images, runs YOLOv5 detection, simulates radar data
│   ├── calibration_projection.py  # Loads calibration matrices, projects LIDAR to image
│   ├── sensor_fusion.py           # Refines YOLO detections using LIDAR data
│   ├── decision_making.py         # Collision avoidance logic
│   └── ...                        
├── main.py                        # 2D Contour-Based Refinement Pipeline
├── main1.py                       # 3D LIDAR-Based Refinement Pipeline
├── requirements.txt               # Required Python packages
└── README.md                      # Project documentation (this file)
```

---

## Workflow Steps

### Step 1: Data Preprocessing & Calibration

- **Load Sensor Data:**  
  Load a specific frame from the camera image directory and the corresponding LIDAR binary file.
  
- **Calibration:**  
  Read calibration files to obtain transformation matrices (Velodyne-to-Camera and Camera-to-Camera). These matrices are used to project the 3D LIDAR points into the 2D camera image plane.

### Step 2: Object Detection

- **YOLOv5 Model:**  
  The pre-trained YOLOv5 model is loaded (via PyTorch Hub) and applied to the camera image to detect objects.  
- **Output:**  
  Initial detections include bounding boxes, class labels (e.g., "car"), and confidence scores.

### Step 3: Sensor Fusion & Refinement

- **2D Contour-Based (main.py):**  
  Uses OpenCV to detect contours in the image to refine YOLO's bounding boxes.
  
- **3D LIDAR-Based (main1.py):**  
  Projects filtered LIDAR points onto the camera image and refines bounding boxes if a sufficient number of LIDAR points fall within them.

### Step 4: Collision Avoidance & Decision Making

- **Decision Logic:**  
  Based on the fused data (detection and sensor measurements), the system compares object distances or refined bounding boxes against a threshold.  
- **Outcome:**  
  A decision (e.g., "Brake" or "Continue") is generated for each object and overlaid on the final image.

### Step 5: Visualization & Output

- **Overlay Visualization:**  
  Final images display:
  - Original YOLO detections (blue boxes)
  - Refined detections (green boxes)
  - Color-coded LIDAR points (indicating distance)
  - Decision text (e.g., "Decision: Continue")
  
- **Output File:**  
  The final annotated image is saved (e.g., `output_lidar_refined_0000000010.png`).

---

## How to Run the Scripts

### 2D Contour-Based Refinement (main.py)

1. **Ensure Data & Dependencies:**  
   Confirm that your KITTI data subset is located in `data/2011_09_26_drive_0048_sync/` and calibration files are in `data/2011_09_26_calib/`.

2. **Execute the Script:**  
   Run:
   ```bash
   python main.py
   ```
3. **Process:**  
   - The script loads a camera frame (default frame number 10).  
   - YOLOv5 detects objects and outputs bounding boxes.  
   - OpenCV-based contour detection refines these boxes.  
   - Collision avoidance logic determines and prints the decision in the console.
   
4. **Output:**  
   The console displays the fused data and decisions. The final annotated image is displayed and saved.

### 3D LIDAR-Based Refinement (main1.py)

1. **Ensure Data & Dependencies:**  
   Calibration files and LIDAR `.bin` files must be in their proper locations within `data/`.

2. **Execute the Script:**  
   Run:
   ```bash
   python main1.py
   ```
3. **Process:**  
   - Loads the same frame (default frame number 10).  
   - YOLOv5 is used for initial object detection.  
   - LIDAR points are loaded, filtered, and projected into the camera image using calibration matrices.  
   - The script refines the YOLO bounding boxes using the projected LIDAR data and then applies collision avoidance logic.
   
4. **Output:**  
   The console prints LIDAR filtering statistics, refined bounding boxes, and collision decisions. Two display windows show the intermediate and final annotated images, and the final image is saved to disk.

---

## Sample Output

When running `python main.py`, you may see console output similar to:

```
Using cache found in C:\Users\abhis/.cache\torch\hub\ultralytics_yolov5_master
YOLOv5s summary: 213 layers, 7225885 parameters, 16.4 GFLOPs
Fused Data with Refined Detections:
{'frame_number': 10, 'detections': [{...}], 'radar': {'distance': 11.18, 'relative_speed': 4.67}}
Decision Actions:
{'class': 'car', 'action': 'Continue'}
...
```

And running `python main1.py` shows additional statistics like:
- Number of LIDAR points before and after filtering
- Number of valid projected points
- Final annotated images displaying bounding boxes, color-coded LIDAR points, and decision text

The saved output image (e.g., `output_lidar_refined_0000000010.png`) includes:
- **Blue boxes**: YOLO detections  
- **Green boxes**: LIDAR-refined bounding boxes  
- **Color-coded dots**: LIDAR points overlaid on the image  
- **Overlay Text**: Collision decision (e.g., "Decision: Continue")
![image](https://github.com/user-attachments/assets/a85da871-5582-4ef8-b075-e00ef7b77ce9)

---


## References

- **KITTI Dataset**: [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)
- **YOLOv5 GitHub Repository**: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- **PyTorch Hub YOLOv5 Usage**: [https://github.com/ultralytics/yolov5/wiki/PyTorch-Hub-Usage](https://github.com/ultralytics/yolov5/wiki/PyTorch-Hub-Usage)
-  Abstract Refer this for the simple end
[ADAS.pdf](https://github.com/user-attachments/files/19409944/ADAS.pdf)

 
  
WHY YOLO?

YOLO (You Only Look Once) is not the only object detection model available. There are many alternatives, including SSD (Single Shot MultiBox Detector), Faster R-CNN, EfficientDet, and DETR. However, we selected a pre-trained YOLO model for this ADAS prototype due to the following reasons:

1. Speed vs. Accuracy Tradeoff

YOLO: One of the fastest object detection models, capable of running in real-time (e.g., YOLOv5 can achieve 30+ FPS on a decent GPU).

Faster R-CNN: Provides higher accuracy but is significantly slower (around 7 FPS), making it unsuitable for real-time ADAS applications.

SSD: Faster than Faster R-CNN but slightly less accurate than YOLO. It’s an alternative, but YOLO's latest versions outperform SSD in both speed and accuracy.

EfficientDet: Highly accurate but computationally heavy. It is optimized for efficiency on GPUs but not for real-time vehicle safety applications.


2. Real-Time Performance for ADAS

ADAS applications require immediate decision-making (e.g., obstacle detection and collision avoidance).

YOLO processes an entire image in a single forward pass, making it significantly faster than models that perform region-based proposals (like Faster R-CNN).

High FPS is critical for ADAS to prevent delayed braking or steering responses.


3. Single-Stage vs. Two-Stage Models

YOLO, SSD: These are single-stage detectors, meaning they directly predict bounding boxes and class probabilities from an image in one pass.

Faster R-CNN: A two-stage detector that first proposes regions and then classifies them, leading to higher accuracy but more latency.

Since ADAS prioritizes real-time reaction over absolute accuracy, single-stage detectors like YOLO are preferred.


4. Ease of Deployment and Pre-Trained Models

YOLO has numerous pre-trained models available in frameworks like PyTorch and TensorFlow.

It can be easily fine-tuned with a custom dataset.

Deployment on edge devices (Jetson Nano, Raspberry Pi, etc.) is easier compared to larger models like EfficientDet.


5. Multi-Object Detection Capability

YOLO effectively detects vehicles, pedestrians, cyclists, road signs, etc., which are crucial for ADAS.

Its anchor-based detection allows handling objects at different scales efficiently.


6. Open-Source and Lightweight Variants

YOLO has lighter versions like YOLOv5 Nano and YOLOv8 Small, which can run on embedded systems with lower power consumption.

EfficientDet and DETR, while accurate, require more computing power, making them less practical for low-latency ADAS applications.



---

Alternatives and When to Use Them:


---

Conclusion: Why YOLO?

Real-time processing ✅

Accurate enough for ADAS applications ✅

Lightweight and easy to deploy ✅

Pre-trained models available ✅


If we needed extreme accuracy for specialized tasks (like medical or industrial inspection), Faster R-CNN or EfficientDet could be considered. But for real-time ADAS applications where reaction time is critical, YOLO is the best choice.
