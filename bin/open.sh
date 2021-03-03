#!/bin/bash
echo "Running Script."
sudo docker run -it --name vision2021 --device=/dev/video0 --runtime nvidia --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" --rm yolov5-cavs bash -c "cd Vision2021/powercell_model/YOLOv5_Trained_Model && git pull && python3 detect.py --weights ./best.pt --source 0"