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

from face_recog.face_detection_dlib import FaceDetectorDlib
from face_recog.face_detection_mtcnn import FaceDetectorMTCNN
from face_recog.face_data_store import FaceDataStore
from face_recog.validators import is_valid_img, path_exists
from face_recog.exceptions import (FaceMissing, InvalidImage, ModelFileMissing, 
            NoFaceDetected, NoNameProvided)
from face_recog.face_detection_opencv import FaceDetectorOpenCV
from face_recog.media_utils import convert_to_dlib_rectangle
import numpy as np
import uuid 
import dlib 
import os
from typing import List, Dict, Tuple

class FaceRecognition:
    keypoints_model_path = 'shape_predictor_5_face_landmarks.dat'
    face_recog_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
    
    def __init__(self, model_loc: str="./models", 
                persistent_data_loc='data/facial_data.json',
                face_detection_threshold:int=0.99,
                face_detector:str='dlib') -> None:

        keypoints_model_path = os.path.join(model_loc, 
                                        FaceRecognition.keypoints_model_path)
        face_recog_model_path = os.path.join(model_loc, 
                                        FaceRecognition.face_recog_model_path)
        if not (path_exists(keypoints_model_path) or path_exists(face_recog_model_path)):
            raise ModelFileMissing
        if face_detector == "opencv":
            self.face_detector = FaceDetectorOpenCV(model_loc=model_loc,
                                                    crop_forehead=True, 
                                                    shrink_ratio=0.2) 
        elif face_detector == "mtcnn":
            self.face_detector = FaceDetectorMTCNN(crop_forehead=True, 
                                                    shrink_ratio=0.2)
        else:
            self.face_detector = FaceDetectorDlib()
        self.face_detection_threshold = face_detection_threshold

        self.keypoints_detector = dlib.shape_predictor(keypoints_model_path)
        self.face_recognizor = dlib.face_recognition_model_v1(face_recog_model_path)
        self.datastore = FaceDataStore(persistent_data_loc=persistent_data_loc)


    def register_face(self, image=None, name:str=None, 
                    bbox:List[int]=None):
        """ Registers a single face"""
        
        if not is_valid_img(image) or name is None:
            raise NoNameProvided if name is None else InvalidImage
        
        image = image.copy()
        face_encoding = None
        
        try:
            if bbox is None:    
                bboxes = self.face_detector.detect_faces(image=image)
                if len(bboxes) == 0:
                    raise NoFaceDetected
                bbox = bboxes[0]
            face_encoding = self.get_facial_fingerprint(image, bbox)
        
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
    

    def save_facial_data(self, facial_data:Dict=None) -> bool:
        if facial_data is not None:
            self.datastore.add_facial_data(facial_data=facial_data)
            return True
        return False

    def get_registered_faces(self) -> List[Dict]:
        return self.datastore.get_all_facial_data()


    def recognize_faces(self, image, threshold:float=0.6, 
                        bboxes:List[List[int]]=None):
        """ Finds a matching face for the give in the input
        image. The input image should be cropped to contain
        only one face and then sent to this method."""
        if image is None:
            return InvalidImage
        image = image.copy()

        if bboxes is None:    
            bboxes = self.face_detector.detect_faces(image=image)
            if len(bboxes) == 0:
                raise NoFaceDetected
        # Load the data of existing registered faces
        # compare using the metric the closest match
        all_facial_data = self.datastore.get_all_facial_data()
        matches = []
        for bbox in bboxes:
            face_encoding = self.get_facial_fingerprint(image, bbox)
            match, min_dist = None, 10000000

            for face_data in all_facial_data:
                dist = self.euclidean_distance(face_encoding, 
                                                face_data['encoding'])
                if dist <= threshold and dist < min_dist:
                    match = face_data
                    min_dist = dist
            # bound box, matched face details, dist from closest match
            matches.append((bbox, match, min_dist))
        return matches


    def get_facial_fingerprint(self, image, bbox:List[int]=None) -> List[float]:
        if bbox is None:
            raise FaceMissing
        # Convert to dlib format rectangle
        bbox = convert_to_dlib_rectangle(bbox)
        # Get the facial landmark coordinates
        face_keypoints = self.keypoints_detector(image, bbox)
        
        # Compute the 128D vector that describes the face in an img identified by
        # shape. In general, if two face descriptor vectors have a Euclidean
        # distance between them less than 0.6 then they are from the same
        # person, otherwise they are from different people. 
        face_encoding = self.get_face_encoding(image, face_keypoints)
        return face_encoding


    def get_face_encoding(self, image, face_keypoints:List):
        encoding = self.face_recognizor\
                    .compute_face_descriptor(image, face_keypoints, 1)
        return np.array(encoding)


    def euclidean_distance(self, vector1:Tuple, vector2:Tuple):
        return np.linalg.norm(np.array(vector1) - np.array(vector2)) 


if __name__ == "__main__":
    ############ Sample Usage and Testing ################
    from face_recog.media_utils import load_image_path

    ob = FaceRecognition(model_loc='models', 
                persistent_data_loc='data/facial_data.json',
                face_detector='dlib')
    img1 = load_image_path('data/sample/1.jpg')
    img2 = load_image_path('data/sample/2.jpg')
    img3 = load_image_path('data/sample/sagar.jpg')
    img4 = load_image_path('data/sample/vidit.jpg')
    img5 = load_image_path('data/sample/sagar2.jpg')
    # data1 = ob.register_face(image=img1, name='Test1')
    # data2 = ob.register_face(image=img2, name='Test2')
    
    # # print(data1)
    # print(data2)

    # print('Match:', ob.euclidean_distance(list(data1['encoding']), list(data2['encoding'])))
    
    # ob.register_face(image=img1, name='Test1')
    # ob.register_face(image=img2, name='Test2')
    # ob.register_face(image=img4, name='Vidit')
    # ob.register_face(image=img3, name='Sagar')

    # fd = FaceDetectorMTCNN()
    # fd2 = FaceDetectorOpenCV()
    # print('FD',fd.detect_faces(img3))
    # print('FD2',fd2.detect_faces(img3))
    
    # print('Attempting face recognition...')
    # match, dist = ob.recognize_face(img5, check_face_count=True)
    # print(match['name'] if match and 'name' in match else '', dist)

    os.remove('data/facial_data.json')


    

