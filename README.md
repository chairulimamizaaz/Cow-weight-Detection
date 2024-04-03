# YOLO Object Detection with Machine Learning Integration

This repository contains code for object detection using YOLO (You Only Look Once) and integrating machine learning models for further analysis. Below are instructions on how to use and customize the detection process.

## Instructions

To run the detection process, you can use the `detect.py` file with different options depending on your source:

1. **Using a Webcam:**
    ```bash
    python detect.py --weights best.pt --source 0 --save-txt 
    ```

2. **Using a Video or Image:**
    ```bash
    python detect.py --weights best.pt --source video.mp4 --save-txt
    ```

    Note: Replace `video.mp4` with the path to your video or image file.

## YOLO Models

The YOLO model files (`best.pt`) are located in the `YOLO Model` folder. There are various versions available:

- `best.pt`: Basic model without upscaling.
- `best.pt(1)`: Model with upscaling.
- `best.pt(2)`: Model with upscaling and an additional class (human).

You can choose the desired version by renaming the file to `best.pt`.

## Machine Learning Models

The machine learning models for distance and body analysis are stored in the `Model machine learning` folder. You can customize the model used in the `detect.py` file:

1. **Distance Model:**
    ```python
    loaded_model = joblib.load("distance_xgb.joblib")  # Line 200
    ```

2. **Body Model:**
    ```python
    loaded_model = joblib.load("body_xgb.joblib")  # Line 204
    ```

## Modifications to `detect.py`

Several modifications have been made to the `detect.py` file:

1. Added object counts and labels for each detected object.
2. Modified coordinate handling to integrate with machine learning.
3. Added loading of machine learning models for distance and body analysis using joblib.
4. Implemented export of detection results to JSON format.
5. Enhanced visualization of bounding boxes with object labels.

The modifications start at line 178 in the `detect.py` file.

## Output and Results

All output records from the `detect.py` process can be found in the `runs/detect` directory. Additionally, machine learning predictions are exported to JSON files for further analysis.

For detailed explanations of the modifications, please refer to the comments in the `detect.py` file.

