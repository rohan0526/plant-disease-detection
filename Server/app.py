import sys
import os

# Ensure Detectron2 is in the Python path
sys.path.insert(0, os.path.abspath('./detectron2'))

from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import json
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Flask app!"

# Ensure the uploads directory exists
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

cfg = get_cfg()

@app.route('/predictPotato', methods=['POST'])
def predict_for_potato():

    # Set up Detectron2 configuration
    
    cfg.merge_from_file('detectron2_model/potato_model/config.yaml')
    cfg.MODEL.WEIGHTS = os.path.join('detectron2_model/potato_model', 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)

    # Load metadata
    with open('detectron2_model/potato_model/train_metadata(1).json', "r") as f:
        loaded_metadata = json.load(f)
    MetadataCatalog.get("my_dataset_val").set(thing_classes=loaded_metadata["thing_classes"])


    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save uploaded file
    filepath = os.path.join('static/uploads/', file.filename)
    file.save(filepath)
    
    # Load the image
    im = cv2.imread(filepath)
    if im is None:
        return jsonify({"error": "Error reading image file"}), 400

    # Get predictions
    outputs = predictor(im)
    instances = outputs["instances"]
    
    pred_classes = instances.pred_classes
    class_names = MetadataCatalog.get("my_dataset_val").get("thing_classes", [])
    
    # Prepare output
    results = {
        "predicted_classes": [class_names[int(cls)] for cls in pred_classes.tolist()],
        "num_classes": len(class_names)
    }

    # Calculate and print the mask areas
    try:
        leaf_index = (pred_classes == 0).nonzero(as_tuple=True)[0]
        blight_index = (pred_classes == 1).nonzero(as_tuple=True)[0]

        leaf_mask = instances.pred_masks[leaf_index]
        blight_mask = instances.pred_masks[blight_index]

        leaf_mask_np = leaf_mask[0].cpu().numpy() if len(leaf_mask) > 0 else np.zeros_like(im[:,:,0])
        blight_mask_np = blight_mask[0].cpu().numpy() if len(blight_mask) > 0 else np.zeros_like(leaf_mask_np)

        leaf_area = leaf_mask_np.sum()
        blight_area = blight_mask_np.sum()

        total_area = leaf_area + blight_area
        blight_ratio = blight_area / total_area if total_area > 0 else 0

        # Define confidence threshold
        CONFIDENCE_THRESHOLD = 0.7  # adjust as needed
        blight_confidence = instances.scores[blight_index].mean().item() if len(blight_index) > 0 else 0

        # Improved blight identification logic
        if blight_ratio < 0.01:
            status = "Healthy"
        elif blight_confidence > CONFIDENCE_THRESHOLD:
            if blight_ratio < 0.1:
                status = "Early Stage Blight"
            elif blight_ratio < 0.3:
                status = "Moderate Blight"
            else:
                status = "Severe Blight"
        else:
            status = "Suspected Blight (Low Confidence)"

        # Add these details to the results
        results.update({
            "leaf_area": int(leaf_area),
            "blight_area": int(blight_area),
            "blight_ratio": float(blight_ratio),
            "blight_confidence": float(blight_confidence),
            "infection_status": status
        })

    except Exception as e:
        results["error"] = f"Error calculating areas: {str(e)}"
    # Visualize the predictions
    v = Visualizer(im[:, :, ::-1],
                   metadata=MetadataCatalog.get("my_dataset_val"),
                   scale=1.0,
                   instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(instances.to("cpu"))
    
    # Save visualization
    vis_path = os.path.join('static/uploads/', 'PotatoVisualization.jpg')
    cv2.imwrite(vis_path, out.get_image()[:, :, ::-1])

    # Construct URL for visualization
    vis_url = f"/static/uploads/PotatoVisualization.jpg"

    results["visualization_url"] = vis_url

    return jsonify(results), 200


@app.route('/predictTomato', methods=['POST'])
def predict_for_tomato():

    # Set up Detectron2 configuration
    cfg.merge_from_file('detectron2_model/tomato_model/config.yaml')
    cfg.MODEL.WEIGHTS = os.path.join('detectron2_model/tomato_model', 'model_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # 5 classes as per the manually set classes
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)

    # Load metadata
    dataset_name = "tomatotraining"  
    thing_classes = ["Tomato Leaf", "Late Blight", "Bacterial Spot", "Early Blight", "Mold"]

    MetadataCatalog.get(dataset_name).set(thing_classes=thing_classes)

    # Check if a file is uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save uploaded file
    filepath = os.path.join('static/uploads/', file.filename)
    file.save(filepath)
    
    # Load the image
    im = cv2.imread(filepath)
    if im is None:
        return jsonify({"error": "Error reading image file"}), 400

    # Get predictions
    outputs = predictor(im)
    instances = outputs["instances"]

    # Initialize the results dictionary
    results = {}

    try:
        # Get predicted class indices and scores
        pred_classes = instances.pred_classes.cpu().numpy()  # Class indices
        scores = instances.scores.cpu().numpy()  # Prediction confidence scores
        class_names = MetadataCatalog.get(dataset_name).get("thing_classes", [])

        # Initialize a list to store detected class names
        detected_class_names = []
        is_infected = False

        # Iterate over predictions and classify as "Infected" or "Healthy"
        for idx, score in zip(pred_classes, scores):
            if idx < len(class_names):
                class_name = class_names[idx]
                detected_class_names.append(class_name)

                # Check if the prediction is for a disease and confidence > 0.3
                if class_name != "Tomato Leaf" and score > 0.3:
                    is_infected = True

        # Final classification based on detected class names
        if not is_infected and "Tomato Leaf" in detected_class_names and len(detected_class_names) == 1:
            status = "Healthy"
        else:
            status = "Infected"

        unique_detected_class_names = set(detected_class_names)

        # Return the classification result
        results["infection_status"] = status
        confidence = 100  # Set confidence arbitrarily for now
        results["confidence"] = confidence

        results.update({
            "disease_detected": list(unique_detected_class_names),
            "blight_confidence": confidence,
            "infection_status": status
        })
        
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

    try:
        v = Visualizer(im[:, :,::-1],
                       metadata=MetadataCatalog.get(dataset_name),
                       scale=1.0,
                       instance_mode=ColorMode.IMAGE_BW)
        out = v.draw_instance_predictions(instances.to("cpu"))

        # Save visualization
        vis_path = os.path.join('static/uploads/', 'TomatoVisualization.jpg')
        cv2.imwrite(vis_path, out.get_image()[:, :, ::-1])

        # Construct URL for visualization
        vis_url = f"/static/uploads/TomatoVisualization.jpg"
        results["visualization_url"] = vis_url

    except Exception as e:
        return jsonify({"error": f"Error during visualization: {str(e)}"}), 500

    # Return the results as JSON
    return jsonify(results), 200



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

