# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
'''Description: Class for Face Recognition related methods.
Main operations: Register and Recognize face.'''
# ===================================================

import cv2
from face_recog.exceptions import ModelFileMissing
from face_recog.face_detection import FaceDetector

class FaceRecognition:
    def __init__(self, model_loc: str="./models") -> None:
        self.face_detector = FaceDetector(model_loc=model_loc)

    def register_face(self, image):
        pass

    def recognize_face(self, image):
        pass

    
        