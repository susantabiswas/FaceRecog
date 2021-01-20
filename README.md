[![HitCount](http://hits.dwyl.io/susantabiswas/FaceRecog.svg)](http://hits.dwyl.io/susantabiswas/FaceRecog)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/susantabiswas/FaceRecog.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/susantabiswas/FaceRecog/context:python)
[![Maintainability](https://api.codeclimate.com/v1/badges/97e039df521b8ecf87c2/maintainability)](https://codeclimate.com/github/susantabiswas/FaceRecog/maintainability)
![Tests](https://github.com/susantabiswas/FaceRecog/workflows/Tests/badge.svg)
[![Build Status](https://travis-ci.com/susantabiswas/FaceRecog.svg?branch=master)](https://travis-ci.com/susantabiswas/FaceRecog)
[![codecov](https://codecov.io/gh/susantabiswas/FaceRecog/branch/master/graph/badge.svg?token=CW7OR72KCW)](https://codecov.io/gh/susantabiswas/FaceRecog)



# Facial Recognition System
This face recognition library is built with ease and customization in mind. There are numerous control parameters to control how you want to use the features, be it face detection, face recognition on videos, or with a webcam. 
<br>
At its core, the facial recognition system uses **Siamese Neural network**. Over the years there have been different architectures published and implemented. The library uses **dlib**'s face recognition model, which is inspired from **ResNet-34** network. The modified ResNet-34 has 29 Convolutional layers. The model achieved 99.38% accuracy on LFW dataset. 

There are 4 different face detectors for usage. Wrappers for video and webcam processing are provided for convenience.<br><br>

## Table of Contents
- [Sample Output](#sample-output)
- [Architecture](#architecture)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [References](#references)

# Sample Output

## Processed Video
<img src="data/media/output.gif"/><br>

## Processed Images

<img src="data/media/1.jpg" height="320" /><img src="data/media/2.jpg" height="320" />
<img src="data/media/7.jpg" height="320" /><img src="data/media/8.jpg" height="320" />
<img src="data/media/3.jpg" height="320" /><img src="data/media/4.jpg" height="320" />
<img src="data/media/9.jpg" height="320" />
<!-- <img src="data/media/5.jpg" height="320" /><img src="data/media/6.jpg" height="320" /> -->



# Architecture
![architecture](data/media/architecture.png)<br>

For face recognition, flow is:

    media -> frame -> face detection -> Facial ROI -> Neural Network -> 128D facial encoding 

These are the major components:
1. **Face Detection**: There are 4 different face detectors with different cropping options.
2. **Face Recognition**: Responsible for handling facial recognition related functionalities like registering facial data etc. 
3. **Storage**: The system provides abstract definitions of cache and persistent storage. For usage, a simple cache using python's native data structure is provided along side a persistent storage system with JSON. If needed the abstract classes can be extended to integrate better storage systems. 
4. **Utilities**: Methods for handling image, video operations, validations, etc.

<br>

# Setup
There are multiple ways to set this up.
### Clone the repo and install dependencies.<br>
```python
git clone https://github.com/susantabiswas/FaceRecog.git
pip install -r requirements.txt
```

### Docker Image
You can pull the docker image for this project and run the code there.<br>
```docker pull susantabiswas/face_recog:latest```

### Dockerfile
You can build the docker image from the docker file present in the repo.

```docker build -t <name> .```


# Project Structure
```
FaceRecog/
├── Dockerfile
├── README.md
├── data/
├── docs/
├── face_recog/
│   ├── exceptions.py
│   ├── face_data_store.py
│   ├── face_detection_dlib.py
│   ├── face_detection_mtcnn.py
│   ├── face_detection_opencv.py
│   ├── face_detector.py
│   ├── face_recognition.py
│   ├── in_memory_cache.py
│   ├── json_persistent_storage.py
│   ├── logger.py
│   ├── media_utils.py
│   ├── persistent_storage.py
│   ├── simple_cache.py
│   └── validators.py
├── models/
│   ├── dlib_face_recognition_resnet_model_v1.dat
│   ├── mmod_human_face_detector.dat
│   ├── opencv_face_detector.pbtxt
│   ├── opencv_face_detector_uint8.pb
│   └── shape_predictor_5_face_landmarks.dat
├── requirements.txt
├── tests/
│   ├── conftest.py
│   ├── test_face_data_store.py
│   ├── test_face_detection_dlib.py
│   ├── test_face_detection_mtcnn.py
│   ├── test_face_detection_opencv.py
│   ├── test_face_recognition.py
│   ├── test_json_persistent_storage.py
│   ├── test_media_utils.py
│   └── test_simple_cache.py
└── video_main.py
```

# Usage

### Face Recognition
Depending on the use case, whether to aim for accuracy and stability or speed etc., you can pick the face detector. Also, there are customization options inside face detectors to decide the facial ROI.


To register a face using a webcam
```python
# Inside project root
import video_main

# You can pick a face detector depending on Acc/speed requirements
face_recognizer = FaceRecognitionVideo(face_detector='dlib')
face_recognizer.register_face_webcam(name="Susanta")
```

To register a face using an image on disk
```python
# Inside project root
import video_main

face_recognizer = FaceRecognitionVideo(face_detector='dlib')
face_recognizer.register_face_path(img_path='data/sample/conan.jpg', name="Conan")
```

To register a face using a loaded image 
```python
# Inside project root
from face_recog.media_utils import load_image_path
from face_recog.face_recognition import FaceRecognition

face_recognizer = FaceRecognition(
                    model_loc="models",
                    persistent_data_loc="data/facial_data.json",
                    face_detector="dlib",
                )
img = load_image_path("data/sample/1.jpg")
# Matches is a list containing information about the matches
# for each of the faces in the image
matches = face_recognizer.register_face(image=img, name=name)
```

Face recognition with a webcam feed
```python
# Inside project root
import video_main

face_recognizer = FaceRecognitionVideo(face_detector='dlib')
face_recognizer.recognize_face_video(video_path=None, \
                                    detection_interval=2, save_output=True, \
                                    preview=True, resize_scale=0.25)
```

Face recognition on a video
```python
# Inside project root
import video_main

face_recognizer = FaceRecognitionVideo(face_detector='dlib')
face_recognizer.recognize_face_video(video_path='data/trimmed.mp4', \
                                    detection_interval=2, save_output=True, \
                                    preview=True, resize_scale=0.25)
```

Face recognition using an image
```python
# Inside project root
from face_recog.media_utils import load_image_path
from face_recog.face_recognition import FaceRecognition

face_recognizer = FaceRecognition(
                    model_loc="models",
                    persistent_data_loc="data/facial_data.json",
                    face_detector="dlib",
                )
img = load_image_path("data/sample/1.jpg")
# Matches is a list containing information about the matches
# for each of the faces in the image
matches = face_recognizer.recognize_faces(
                image=img, threshold=0.6
            )
```


There are 4 face detectors namely dlib (HOG, MMOD), MTCNN, OpenCV (CNN). 
All the face detectors are based on a common abstract class and have a common detection interface **detect_faces(image)**.

```python
# import the face detector you want, it follows absolute imports
from face_recog.media_utils import load_image_path
from face_recog.face_detection_dlib import FaceDetectorDlib

face_detector = FaceDetectorDlib(model_type="hog")
# Load the image in RGB format
image = load_image_path("data/sample/1.jpg")
# Returns a list of bounding box coordinates
bboxes = face_detector.detect_faces(image)
```



# References
The awesome work Davis E. King has done: 
http://dlib.net/cnn_face_detector.py.html, 
https://github.com/davisking/dlib-models<br>
You can find more about MTCNN from here: https://github.com/ipazc/mtcnn
