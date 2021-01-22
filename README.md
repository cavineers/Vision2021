# 2021_ObjectDetection_Vision
2021 FIRST FRC Galactic Search Mission Vision Code. This code will be able to run real time object detection for the playing field using a RPI (4) and TensorFlow Lite. All the resources and vision code shall be updated as more information becomes available.

# YOLOv5 Object Detection Information / Docs
YOLOv5 is an AI object detection library used for real time object detection.

## Resources
[Colab copy where YOLOv5 models can be trained](https://colab.research.google.com/drive/1HlhGHEA7LSkETzBbx-9scGilpzTi1-sT?usp=sharing)\
[YOLOv5 Information (training + getting data)](https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/)\
[Other YOLOv5 Information](https://medium.com/towards-artificial-intelligence/yolo-v5-is-here-custom-object-detection-tutorial-with-yolo-v5-12666ee1774e)

## Usage
**NOTE: ALL Packages and Modules Sit In the Virtual Environment using virtualenv. To run any commands for this repo you must enter the venv.** FROM THE YOLOv5_trained_model DIRECTORY Type `source venv/bin/activate` to start the environmental variable

The files in the powercell_model/YOLOv5_Trained_Model directory are all the trained ML models. It consists of the data.yaml, custom_yolov5s.yaml, and a best.pt that is the trained model file.

*Note: Roboflow is used for the creation of the yolov5 format.*

To get (or update) the trained model:
1. collect training images (pictures of what you want to detect)
2. create a roboflow account and create a roboflow dataset following the prompts
3. after creating a dataset and labeling the images in roboflow, generate the set in the YOLOv5 format and select the view code option on the download page.
4. open the first resource link (colab) and plug the link you copied into the lab (where prompted in the all caps comment)
5. follow the colab until the end
6. download the data.yaml, custom_yolov5s.yaml, and a best.pt files from the menu in the left side of the screen.
7. push those files to the repository

To test against images clone this repository and then put the testing images into the test_imgs dir and type `python3 detect.py --weights ./best.pt  --source ./test_imgs` in a terminal to run a detection. It will output the resulted images into the runs directory.\
To run against a webcam in real time run this command `python3 detect.py --weights ./best.pt  --source 0` changing the source to be 0.

### To create a venv (Using python 3.8 since 3.9 does not work yet with certain packages)
1. download virtualenv with pip3
2. run `python3 -m venv /path/to/new/virtual/environment`
3. enter the venv with `source venv/bin/activate`
4. install the required dependancies in the virtual environment using the requirments.txt file: `pip3 install -qr requirements.txt`


# DEPRECIATED temporarily... TensorFlow Items Below (*Depreciated* under new arch built on Yolov5)
## Resources / Research
**Note:** this list will be updated as more resources become available...

[Vision on TensorFlow Lite](https://www.tensorflow.org/lite/models/object_detection/overview#model_customization)\
[TensorFlow Models GitHub Repository](https://github.com/tensorflow/models)\
[Tranfer Learning with TensorFlow Video Resource](https://www.coursera.org/lecture/device-based-models-tensorflow/transfer-learning-with-tflite-y7OPK)\
[Already Trained TensorFlow Models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#pixel4-edge-tpu-models)\
[Custom ML Model Guide](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)

## Methods
1. We will be using a method called transfer learning to use a previously trained TensorFlow Model to fit our specific requirments.
2. We can also train a custom model from scratch but that would require more work than the above option but still one that is available.

### Credits
*Roboflow*
*YOLOv5*
*TensorFlow*
