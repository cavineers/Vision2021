#!/bin/bash  
echo "Running Script." 
sudo docker run -it --device=/dev/video0 --runtime nvidia --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --rm yolov5-cavs
git pull
cd Vision2021/powercell_model/YOLOv5_Trained_Model
echo "Starting detect.py"
python3 detect.py --weights ./best.pt --source 0