#!/bin/bash  
echo "Running..."
cd Documents/GitHub/2021_ObjectDetection_Vision/powercell_model/YOLOv5_Trained_Model
source venv/bin/activate
python3 detect.py --weights ./best.pt --source 0 --dev t --headless True