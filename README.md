# PolyTrack_v1

## Introduction
This code is related to the paper "Towards Computer Vision and Deep Learning facilitated Pollination Monitoring for Agriculture" submitted to 2nd International Workshop and Prize Challenge on Agriculture-Vision: Challenges & Opportunities for Computer Vision in Agriculture (AgriVision).
 
PolyTrack is designed to track insect pollinators in complex dynamic environments. It uses a combination of foreground-background segmentation (KNN background subtractor) and deep learning-based detection (YOLOv4) for tracking. 

## Dependancies

Dependancies related to this code is provided in requirements-cpu.txt and requirements-gpu.txt files.

## Pre-trained weights for YOLOv4

Pre-trained weights for YOLOv4 can be downloaded from here. 

Rename the weights file to custome.weights and copy and paste it into the "data" folder of this repository.

Use the following commands to convert the darkflow weights to Tensorflow. The pre-trained weights were trained on honeybee and strawberry flower images. Please make sure "./data/classes/custom.name" file containes correct names of the classes (i.e. honeybee and flower)
 
```
python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom-416 --input_size 416 --model yolov4 
```

## Running the code

Code related to the core funtionality of the PolyTrack algorithm is in the folder "polytrack" of this repisotary.

Input and output directories of videos can be specified in file "./polytrack/config.py". User has the option of specifing a single input video or collection of videos. Parameters related to video processing and the output files can be adjusted and declared in the config file. Please refer to the documentation of the config file for more information.

After declaring relevent parameters, navigate to the root folder of the repository and run use the following command to run PolyTrack.
```
python PolyTrack.py 
```

 
## References
 
The YOLOv4 component of this repository was adopted from [darknet repository](https://github.com/AlexeyAB/darknet) by AlexeyAB and [yolov4-custom-functions](https://github.com/theAIGuysCode/yolov4-custom-functions) by the AIGuysCode.
