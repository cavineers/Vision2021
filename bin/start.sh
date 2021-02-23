#!/bin/bash  
echo "Running YOLOv5 Trained Model. This script may take a few minutes." 
echo "Pulling Docker CUDA scripts." 
sudo docker pull nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3 
sudo docker sudo docker run -it --device=/dev/video0 --runtime nvidia --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3
echo "Running Docker Container..."  
echo "Build Success. Starting GitHub image."
git clone https://github.com/cavineers/Vision2021.git
cd Vision2021/powercell_model/YOLOv5_Trained_Model
echo "Updating dependancies and scripts.."
python3 -m pip install --upgrade pip, python3 -m pip install --upgrade setuptools, pip install -r requirements.txt
echo "Creation of image success. Committing Changes."
echo "Open another terminal and run 'sudo docker ps -a' and get the id of this container and then run 'sudo docker commit [CONTAINER ID] yolov5-cavs'"