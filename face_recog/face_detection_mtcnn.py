# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
'''Description: Class for face detection. Uses a MTCNN 
based neural network to get the bounding box coordinates 
for a human face.

Usage: python -m face_recog.face_detection_mtcnn

You can install mtcnn using PIP by typing "pip install mtcnn"
Ref: https://github.com/ipazc/mtcnn
'''
# ===================================================

from face_recog.face_detection import FaceDetectorOPENCV
from face_recog.validators import is_valid_img
from face_recog.exceptions import InvalidImage
import os  
from typing import List
from face_recog.face_detector import FaceDetector
from mtcnn import MTCNN
import cv2

class FaceDetectorMTCNN(FaceDetector):
    def __init__(self):
        try:
            # load the model
            self.face_detector = MTCNN()
        except Exception as e:
            raise e


    def detect_faces(self, image, 
                    conf_threshold: float=0.7)->List[List[int]]:
        if image is None:
            return []
        
        if not is_valid_img(image):
            raise InvalidImage
        # convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Do a forward propagation with the blob created from input img
        detections = self.face_detector.detect_faces(image)
        # Bounding box coordinates of faces in image
        bboxes = []
        for idx, detection in enumerate(detections):
            conf = detection['confidence']
            if conf >= conf_threshold:
                x, y, w, h = detection['box']
                bboxes.append([x, y, x + w, y + h])

        return bboxes


    def __repr__(self):
        return "FaceDetectorMTCNN"


if __name__ == "__main__":
    # Sample Usage
    ob = FaceDetectorMTCNN()
    img = cv2.imread('data/sample/1.jpg')

    # import numpy as np
    # img = np.zeros((100,100,5), dtype='float32')
    bbox = ob.detect_faces(img)
    print(bbox)
    print(ob)