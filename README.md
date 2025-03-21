Below is a **README.md** example describing your two main scripts—**main.py** (using contour-based 2D refinement) and **main1.py** (using 3D LIDAR projection and calibration). Feel free to adjust folder names, file names, or descriptions to match your exact setup.

---

# ADAS & AI/ML Prototype

This repository demonstrates an **Advanced Driver Assistance System (ADAS)** prototype using the **KITTI dataset** for **object detection**, **sensor fusion**, and **collision avoidance**. Two different main scripts showcase **2D contour-based refinement** (main.py) and **3D LIDAR-based refinement** (main1.py).

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Prerequisites & Installation](#prerequisites--installation)  
3. [Data & Calibration Files](#data--calibration-files)  
4. [Key Features](#key-features)  
5. [Project Structure](#project-structure)  
6. [How to Run (2D Contour-Based)](#how-to-run-2d-contour-based)  
7. [How to Run (3D LIDAR-Based)](#how-to-run-3d-lidar-based)  
8. [References](#references)

---

## Project Overview

1. **Camera & LIDAR Data**  
   - Loads camera frames and LIDAR `.bin` files from the KITTI dataset.  
   - Uses calibration files to align LIDAR points with the camera image plane (for the 3D demonstration).

2. **Object Detection**  
   - **YOLOv5** (via PyTorch Hub) is used to detect objects (cars, pedestrians, etc.) in each frame.

3. **Refinement Approaches**  
   - **2D Contour-Based** (main.py): Uses OpenCV contour detection to refine bounding boxes in the image plane.  
   - **3D LIDAR-Based** (main1.py): Projects LIDAR points into the image plane, checks which points fall within YOLO bounding boxes, and refines boxes accordingly.

4. **Collision Avoidance**  
   - A simple decision-making module checks each object’s distance or bounding-box position to decide whether to **“Brake”** or **“Continue.”**

---

## Prerequisites & Installation

1. **Clone/Download** this repository.  
2. **Install Dependencies** with:
   ```bash
   pip install -r requirements.txt
   ```
3. **KITTI Dataset** (subset):  
   - Place `2011_09_26_drive_0048_sync` and the calibration files (`2011_09_26_calib`) in the `data` folder as shown below.

---

## Data & Calibration Files

- **KITTI Data**  
  - `data/2011_09_26_drive_0048_sync/` containing images (`image_02/`) and LIDAR `.bin` files (`velodyne_points/`).  
- **Calibration**  
  - `data/2011_09_26_calib/2011_09_26/` containing `calib_velo_to_cam.txt` and `calib_cam_to_cam.txt` for projecting LIDAR points.

---

## Key Features

- **Object Detection** with YOLOv5  
- **2D Contour Refinement** or **3D LIDAR Refinement**  
- **Collision Avoidance Logic**  
- **Visualization** of bounding boxes, color-coded LIDAR points, and decision overlays

---

## Project Structure

```
ADAS_Project/
├── data/
│   ├── 2011_09_26_calib/
│   └── 2011_09_26_drive_0048_sync/
├── src/
│   ├── data_preprocessing.py
│   ├── decision_making.py
│   ├── sensor_fusion.py
│   ├── calibration_projection.py
│   └── ...
├── main.py        # 2D contour-based refinement
├── main1.py       # 3D LIDAR-based refinement
├── requirements.txt
└── README.md
```

---

## How to Run (2D Contour-Based)

**File:** `main.py`  
**Method:** Uses OpenCV contour detection to refine bounding boxes purely in 2D.

1. **Ensure Data & Dependencies**  
   - Confirm your KITTI subset is in `data/2011_09_26_drive_0048_sync/`.  
   - Run `pip install -r requirements.txt` if not already done.

2. **Execute**  
   ```bash
   python main.py
   ```
3. **Process**  
   - Loads a frame (`frame_number = 10` by default).  
   - Performs YOLO detection to get initial bounding boxes.  
   - Applies **contour-based refinement** (OpenCV) on the camera image.  
   - Runs collision avoidance logic.  
   - Prints the final bounding boxes and decisions in the console.

4. **Output**  
   - **Console**: Fused data with refined detections and decisions (e.g., "Brake" or "Continue").  
   - No LIDAR-based 3D projection in this approach—just 2D image analysis.

---

## How to Run (3D LIDAR-Based)

**File:** `main1.py`  
**Method:** Uses calibration files and LIDAR projection to refine bounding boxes.

1. **Ensure Data & Dependencies**  
   - Calibration files (`calib_velo_to_cam.txt`, `calib_cam_to_cam.txt`) in `data/2011_09_26_calib/`.  
   - LIDAR `.bin` files in `data/2011_09_26_drive_0048_sync/velodyne_points/data/`.

2. **Execute**  
   ```bash
   python main1.py
   ```
3. **Process**  
   - Loads the same frame (`frame_number = 10` by default).  
   - Performs YOLO detection on the camera image.  
   - **Projects LIDAR points** into the image plane using calibration matrices.  
   - Refines bounding boxes if enough LIDAR points fall inside the YOLO boxes.  
   - Applies collision avoidance based on LIDAR distance.  
   - Overlays color-coded LIDAR points on the final image.

4. **Output**  
   - **Console**: LIDAR filtering stats, YOLO detections, refined bounding boxes, collision decisions.  
   - **Windows**:  
     - One with bounding boxes (blue for YOLO, green for refined).  
     - One with color-coded LIDAR points overlaid and a decision text (e.g., "Decision: Continue").  
   - **Saved Image**: `output_lidar_refined_0000000010.png` (or similar).

---

## References

- **KITTI Dataset**: [http://www.cvlibs.net/datasets/kitti/](http://www.cvlibs.net/datasets/kitti/)  
- **YOLOv5**: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)  
- **PyTorch Hub YOLOv5**: [https://github.com/ultralytics/yolov5/wiki/PyTorch-Hub-Usage](https://github.com/ultralytics/yolov5/wiki/PyTorch-Hub-Usage)

---

This completes the overview of how to run both **2D contour-based** (`main.py`) and **3D LIDAR-based** (`main1.py`) ADAS prototypes. Each script demonstrates a different approach to refining YOLO detections, ultimately guiding a simple collision avoidance system.
