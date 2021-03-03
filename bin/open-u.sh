#!/bin/bash  
echo "Running Script."
sleep 34
echo "Starting Vision Systems."
sudo docker run --name vision2021 --device=/dev/video0 --env="DISPLAY" --runtime nvidia --net=host --rm yolov5-cavs bash -c "cd Vision2021/powercell_model/YOLOv5_Trained_Model && python3 detect.py --weights ./best.pt --source 0 --dev t"
