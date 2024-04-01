To run detection we can run it in the detect.py file, example command to run a program detect.py:
1. with camera(webcam) : python detect.py --weights best.pt --source 0 --save-txt 
2. with video/image : python detect.py --weights best.pt --source video.mp4 --save-txt (video.mp4 can be customized with the video/image file name)



For the best.pt file or object detection model YOLO, the file is in the YOLO Model folder (yolov5/YOLO Model)
Various best.pt files
1. best.pt: without upscaling,
2. best.pt(1): with upscale,
3. best.pt(2): with upscale + additional class (human).

To use it, please change the file name to best.pt if you want to use which version



To detect you can set which machine learning will be used in the detect.py file, you can change the machine learning model file in the machine learning model folder (yolov5/Model machine learning/Model) in this yolo file
you can change here:
1. Distance : loaded_model = joblib.load("distance_xgb.joblib") (line 200)
2. Body : loaded_model = joblib.load("body_xgb.joblib") (line 204)



Notes: 
1. All output records detect.py can be seen in the detect file (yolov5-master\runs\detect) 
2. We have provided machine learning files for distance and body by exporting the joblib model which is in the machine learning model folder ((yolov5/Model machine learning/Model)



The following is an explanation of the detect.py modifications that we have made:
1. modify by adding the number per object detected and adding object labels to how many
2. modified it so that the coordinates in Yolo can be read so that it can enter machine learning
3. modified it by adding a load model for the machine learning distance regressor and body regressor using joblib
4. modified it so that there is an export with json of the prediction results
5. modify it so you can see the detection results in the label bounding box
6. Modification starts at line 178 in file detect.py

Following are code the modifications:
```python
    obj_counts = {}  # Dictionary to store counts of each detected object type
    current_object_type = None  # Variable to track the current object type (not used)
    
    # Iterate over detections in reverse order
    for *xyxy, conf, cls in reversed(det):
        obj_class = names[int(cls)]  # Map class index to class name
    
        # Check if the object class has been encountered before
        if obj_class not in obj_counts:
            obj_counts[obj_class] = 1  # Initialize count
        else:
            obj_counts[obj_class] += 1  # Increment count
    
        obj_count = obj_counts[obj_class]  # Get count of current object class
        print(obj_count)  # Print the count
    
        if save_txt:
            # Extract bounding box coordinates
            xywh = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
            x_min, y_min, x_max, y_max = xyxy
    
            # Extract bounding box properties
            x = (x_min + x_max) / 2
            y = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min
    
            # Load models machine learning (distance & body) and make predictions
            loaded_model_distance = joblib.load("distance_xgb.joblib")
            new_data_distance = np.array([[x, y, w, h]])
            distance = loaded_model_distance.predict(new_data_distance)[0]
    
            loaded_model_body = joblib.load("body_xgb.joblib")
            new_data_body = np.array([[x, y, w, h, distance]])
            prediction_body = loaded_model_body.predict(new_data_body)[0]
    
            # Calculate weight with winter formula
            berat = ((prediction_body[0] ** 2) * prediction_body[1]) / 10840
    
            # Print information about the detected object
            print(f"{obj_class} {obj_count}: berat {berat:.2f} kg")
    
            # Save information to a JSON file
            import json
            data = {
                "obj_class": str(obj_class),
                "Obj_count": int(obj_count),
                "Berat": float(berat),
            }
            json_data = json.dumps(data, indent=4)
            with open(f'{txt_path}.json', 'a') as f:
                f.write(json_data + '\n')
    
        # Visualize bounding box on image if necessary
        if save_img or save_crop or view_img:
            c = int(cls)
            label = f"{obj_class} {obj_count}: berat {berat:.2f} kg"
            annotator.box_label(xyxy, label, color=colors(c, True))
    
        # Save cropped region as an image if necessary
        if save_crop:
            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
    
    # If current_object_type is not None, print information about the last encountered object type
    if current_object_type is not None:
        print(f"{obj_class} {obj_count}: berat {berat[0]:.2f} kg")
