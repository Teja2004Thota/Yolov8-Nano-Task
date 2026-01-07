# Overview
This project implements a real-time object detection system using YOLOv8 Nano and Python.
It supports video file upload and live webcam detection, performs object detection in real time, logs detections, measures performance, and generates detailed reports.

The system is designed to be user-friendly, efficient, and interview-ready, focusing on inference and deployment rather than model training.

## Objectives
1.	Perform real-time object detection using a pre-trained YOLO model
2.	Support both video file input and webcam input
3.	Display bounding boxes with confidence scores
4.	Log detection details to a CSV file
5.	Measure and report FPS performance
6.	Generate a detailed TXT performance report with statistics

## Tech Stack
    •	Language: Python 3.8+
    •	Model: YOLOv8 Nano (yolov8n.pt)
    •	Libraries:
        o	OpenCV
        o	Ultralytics
        o	Pandas
        o	Tkinter (file chooser)
        o	NumPy (indirect via dependencies)

## How to Run
        1.	Install Dependencies
                pip install numpy==1.26.4 opencv-python ultralytics pandas
        2.	Run the Script
                python object_detection.py
        3.	Select Input Type
                1 → Choose a video file
                2 → Use webcam
## 📝 Note on Output Files
Before running the script for the first time:

- `detection_logs.csv` does not exist or is empty  
- `performance_report.txt` does not exist or is empty  

These files are **automatically generated and populated after execution**, once object detection is completed.  
No manual file creation is required.


# Confidence & IoU Handling
    •	Confidence Threshold: Applied to filter low-confidence detections
    •	IoU Threshold: Handled internally by YOLOv8 during Non-Maximum Suppression (NMS)
    •	Thresholds can be tuned at inference time for better precision or recall
