# Polytrack_v1

> ### This repository is archived
>**This repository is no longer being updated. For the latest version of the software with ongoing support and frequent updates, please refer to the [Polytrack repository](https://github.com/malikaratnayake/Polytrack)**.

## Introduction
This code is related to the paper "Towards Computer Vision and Deep Learning facilitated Pollination Monitoring for Agriculture" submitted to 2nd International Workshop and Prize Challenge on Agriculture-Vision: Challenges & Opportunities for Computer Vision in Agriculture (AgriVision).
 
Polytrack is designed to track insect pollinators in complex dynamic environments. It uses a combination of foreground-background segmentation (KNN background subtractor) and deep learning-based detection (YOLOv4) for tracking. 

## Dependencies

Dependencies related to this code is provided in requirements-cpu.txt and requirements-gpu.txt files.

## Pre-trained weights for YOLOv4

Pre-trained weights for YOLOv4 can be downloaded from [here](https://drive.google.com/drive/folders/1-FWctW8msxKQvdj8Dbq5PMatMDT8oLLe?usp=sharing). 

Rename the weights file to custom.weights and copy and paste it into the "data" folder of this repository.

Use the following commands to convert the darkflow weights to Tensorflow. The pre-trained weights were trained on honeybee and strawberry flower images. Please make sure "./data/classes/custom.name" file contains the correct names of the classes (i.e. honeybee and flower)
 
```
python save_model.py --weights ./data/custom.weights --output ./checkpoints/custom-416 --input_size 416 --model yolov4 
```

## Running the code

Code related to the core functionality of the Polytrack algorithm is in the folder "polytrack" of this repository.

Input and output directories of videos can be specified in the file "./polytrack/config.py". The user has the option of specifying a single input video or collection of videos. Parameters related to video processing and the output files can be adjusted and declared in the config file. Please refer to the documentation of the config file for more information.

After declaring relevant parameters, navigate to the root folder of the repository and run use the following command to run Polytrack.
```
python PolyTrack.py 
```

## Contact

If there are any inquiries, please don't hesitate to contact me at Malika DOT Ratnayake AT monash DOT edu.
 
## References
 
The YOLOv4 component of this repository was adopted from [darknet repository](https://github.com/AlexeyAB/darknet) by AlexeyAB and [yolov4-custom-functions](https://github.com/theAIGuysCode/yolov4-custom-functions) by the AIGuysCode.
