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

from face_recog.face_data_store import FaceDataStore
from face_recog.validators import is_valid_img
import cv2
from face_recog.exceptions import (InvalidImage, ModelFileMissing, 
            NoFaceDetected, MultipleFacesDetected, NoNameProvided)
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
    
    def __init__(self, model_loc: str="./models", 
                persistent_data_loc='data/facial_data.json',
                face_detection_threshold:int=0.8) -> None:
        keypoints_model_path = os.path.join(model_loc, 
                                        FaceRecognition.keypoints_model_path)
        face_recog_model_path = os.path.join(model_loc, 
                                        FaceRecognition.face_recog_model_path)

        self.face_detector = FaceDetectorOpenCV(model_loc=model_loc)
        self.face_detection_threshold = face_detection_threshold

        self.keypoints_detector = dlib.shape_predictor(keypoints_model_path)
        self.face_recognizor = dlib.face_recognition_model_v1(face_recog_model_path)
        self.datastore = FaceDataStore(persistent_data_loc=persistent_data_loc)


    def register_face(self, image=None, name=None, check_face_count:bool=True):
        """ Registers a single face"""
        
        if image is None or name is None:
            raise NoNameProvided if name is None else InvalidImage
        image = image.copy()
        face_encoding = None
        
        try:
            face_encoding = self.get_facial_fingerprint(image, 
                                        check_face_count=check_face_count)
        
        
            # Convert the numpy array to normal python float list
            # to make json serialization simpler
            facial_data = {
                "id": str(uuid.uuid4()),
                "encoding": tuple(face_encoding.tolist()),
                "name": name
            }
            # save the encoding with the name
            self.save_facial_data(facial_data)
            print('[INFO] Face registered with name: {}'.format(name))
        except Exception as exc:
            raise exc
        return facial_data
    

    def save_facial_data(self, facial_data=None):
        if facial_data is not None:
            self.datastore.add_facial_data(facial_data=facial_data)
            return True
        return False

    def get_registered_faces(self):
        return self.datastore.get_all_facial_data()


    def recognize_face(self, image, threshold=0.6, 
                        check_face_count:bool=True):
        """ Finds a matching face for the give in the input
        image. The input image should be cropped to contain
        only one face and then sent to this method."""
        if image is None:
            return InvalidImage
        image = image.copy()
        
        # Load the data of existing registered faces
        # compare using the metric the closest match
        all_facial_data = self.datastore.get_all_facial_data()
        face_encoding = self.get_facial_fingerprint(image, 
                                        check_face_count=check_face_count)
        match, min_dist = None, 10000000

        for face_data in all_facial_data:
            dist = self.euclidean_distance(face_encoding, 
                                            face_data['encoding'])
            if dist < min_dist:
                match = face_data
                min_dist = dist
        return match, min_dist

    def get_facial_fingerprint(self, image, check_face_count:bool=True):
        bboxes = [1, 1, image.shape[1]-1, image.shape[0]-1]
        # If the input image is already a cropped ROI, no need to check 
        # for number of faces in image
        if check_face_count:
            # Face detection: If there are multiple faces detected
            # we dont register and throw an exception
            try:
                bboxes = self.face_detector.detect_faces(image=image,
                                        conf_threshold=self.face_detection_threshold)
            except Exception as exc:
                raise exc
            # In order to register a person, we should make sure
            # only one face is present
            if len(bboxes) != 1:
                raise NoFaceDetected if not len(bboxes) else MultipleFacesDetected
            bboxes = bboxes[0]

        # Convert the image back to RGB to feed to dlib based models
        try:
            image = convert_to_rgb(image=image)
        except InvalidImage:
            raise InvalidImage

        # Convert to dlib format rectangle
        bbox = convert_to_dlib_rectangle(bboxes)
        # Get the facial landmark coordinates
        face_keypoints = self.keypoints_detector(image, bbox)

        # Compute the 128D vector that describes the face in an img identified by
        # shape. In general, if two face descriptor vectors have a Euclidean
        # distance between them less than 0.6 then they are from the same
        # person, otherwise they are from different people. 
        face_encoding = self.get_face_encoding(image, face_keypoints)

        return face_encoding

    def get_face_encoding(self, image, face_keypoints):
        encoding = self.face_recognizor\
                    .compute_face_descriptor(image, face_keypoints)
        return np.array(encoding)


    def euclidean_distance(self, vector1, vector2):
        return np.linalg.norm(vector1 - vector2) 


if __name__ == "__main__":
    from face_recog.media_utils import load_image_path

    ob = FaceRecognition(model_loc='models', 
                persistent_data_loc='data/facial_data.json')
    img1 = load_image_path('data/sample/1.jpg')
    img2 = load_image_path('data/sample/2.jpg')

    data1 = ob.register_face(image=img1, name='Test1')
    # data2 = ob.register_face(image=img2, name='Test2')
    
    # print(data1)
    # print(data2)

    # print('Match:', ob.euclidean_distance(data1['encoding'], data2['encoding']))
    match, dist = ob.recognize_face(img2)
    print(match['name'], dist)

    os.remove('data/facial_data.json')
