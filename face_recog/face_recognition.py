# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
'''Description: Class for Face Recognition related methods.
Main operations: Register and Recognize face.

Usage: python -m face_recog.face_recognition

dlib model files can be downloaded from:
http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"

https://github.com/davisking/dlib-models
'''
# ===================================================

from face_recog.validators import is_valid_img
import cv2
from face_recog.exceptions import (InvalidImage, ModelFileMissing, 
            NoFaceDetected, MultipleFacesDetected)
from face_recog.face_detection_opencv import FaceDetectorOpenCV
from face_recog.media_utils import (convert_to_rgb,
                    convert_to_dlib_rectangle)
import numpy as np
import uuid 
import dlib 
import sys
import os

class FaceRecognition:
    keypoints_model_path = 'shape_predictor_5_face_landmarks.dat'
    face_recog_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
    
    def __init__(self, model_loc: str="./models") -> None:
        keypoints_model_path = os.path.join(model_loc, 
                                        FaceRecognition.keypoints_model_path)
        face_recog_model_path = os.path.join(model_loc, 
                                        FaceRecognition.face_recog_model_path)

        self.face_detector = FaceDetectorOpenCV(model_loc=model_loc)
        self.keypoints_detector = dlib.shape_predictor(keypoints_model_path)
        self.face_recognizor = dlib.face_recognition_model_v1(face_recog_model_path)


    def register_face(self, image, name):
        """ Registers a single face"""
        image = image.copy()

        # Face detection: If there are multiple faces detected
        # we dont register and throw an exception
        try:
            bboxes = self.face_detector.detect_faces(image=image)
        except Exception as exc:
            raise exc

        # In order to register a person, we should make sure
        # only one face is present
        if len(bboxes) != 1:
            raise NoFaceDetected if not len(bboxes) else MultipleFacesDetected
        # Convert the image back to RGB to feed to dlib based models
        try:
            image = convert_to_rgb(image=image)
        except InvalidImage:
            raise InvalidImage

        # Convert to dlib format rectangle
        bbox = convert_to_dlib_rectangle(bboxes[0])
        # Get the facial landmark coordinates
        face_keypoints = self.keypoints_detector(image, bbox)

        # Compute the 128D vector that describes the face in an img identified by
        # shape. In general, if two face descriptor vectors have a Euclidean
        # distance between them less than 0.6 then they are from the same
        # person, otherwise they are from different people. 
        face_encoding = self.get_face_encoding(image, face_keypoints)
        
        # Convert the numpy array to normal python float list
        # to make json serialization simpler
        facial_data = {
            "id": uuid.uuid4(),
            "encoding": tuple(face_encoding.tolist()),
            "name": name
        }
        # save the encoding with the name
        self.save_facial_data(facial_data)
        
        return facial_data
    

    def save_facial_data(self, facial_data):
        pass


    def recognize_face(self, image, threshold=0.6):
        pass


    def get_face_encoding(self, image, face_keypoints):
        encoding = self.face_recognizor\
                    .compute_face_descriptor(image, face_keypoints)
        return np.array(encoding)


    def euclidean_distance(self, vector1, vector2):
        return np.linalg.norm(vector1 - vector2) 


if __name__ == "__main__":
    from face_recog.media_utils import load_image_path

    ob = FaceRecognition(model_loc='models')
    img1 = load_image_path('data/sample/1.jpg')
    img2 = load_image_path('data/sample/2.jpg')

    print(img1.shape)
    # data1 = ob.register_face(image=img1, name='Test1')
    data2 = ob.register_face(image=img2, name='Test2')

    # print(data1)
    print(data2)

    # print('Match:', ob.euclidean_distance(data1['encoding'], data2['encoding']))

