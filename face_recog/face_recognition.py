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
from face_recog.validators import is_valid_img
import cv2
from face_recog.exceptions import (FaceMissing, InvalidImage, ModelFileMissing, 
            NoFaceDetected, MultipleFacesDetected, NoNameProvided)
from face_recog.face_detection_opencv import FaceDetectorOpenCV
from face_recog.media_utils import (convert_to_rgb,
                    convert_to_dlib_rectangle, draw_annotation, draw_bounding_box, get_facial_ROI)
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
                face_detection_threshold:int=0.99,
                face_detector:str='dlib') -> None:

        keypoints_model_path = os.path.join(model_loc, 
                                        FaceRecognition.keypoints_model_path)
        face_recog_model_path = os.path.join(model_loc, 
                                        FaceRecognition.face_recog_model_path)

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


    def register_face(self, image=None, name=None, 
                    bbox=None):
        """ Registers a single face"""
        
        if not is_valid_img(image) or name is None:
            raise NoNameProvided if name is None else InvalidImage
        
        image = image.copy()
        face_encoding = None
        
        try:
            if bbox is None:    
                bboxes = self.face_detector.detect_faces(image=image)
                if len(bboxes) == 0:
                    raise FaceMissing
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
    

    def save_facial_data(self, facial_data=None):
        if facial_data is not None:
            self.datastore.add_facial_data(facial_data=facial_data)
            return True
        return False

    def get_registered_faces(self):
        return self.datastore.get_all_facial_data()


    def recognize_faces(self, image, threshold=0.6, bboxes=None):
        """ Finds a matching face for the give in the input
        image. The input image should be cropped to contain
        only one face and then sent to this method."""
        if image is None:
            return InvalidImage
        image = image.copy()

        if bboxes is None:    
            bboxes = self.face_detector.detect_faces(image=image)
        #     bbox = bboxes[0]
        #     """ Put exception here"""
        #     if len(bboxes) == 0:
        #         print('&&&&&&&&&&&&&&&&')
        #         return
        # print('BBOX', bbox)
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
                print('dist:', dist, 'name:', face_data['name'])
                if dist <= threshold and dist < min_dist:
                    match = face_data
                    min_dist = dist
            # bound box, matched face details, dist from closest match
            matches.append((bbox, match, min_dist))
        return matches


    def face_encodings(self, image, bboxes=None):
        """ Put exception """
        # if bboxes is None:
        #     print('[ERROR]$$$$$$$$$$$$$$3')
        #     return
        # print('what', bboxes)
        # bbox = [1, 1, image.shape[1]-1, image.shape[0]-1]
        
        # If the input image is already a cropped ROI, no need to check 
        # for number of faces in image
        if True:
            # Face detection: If there are multiple faces detected
            # we dont register and throw an exception
            try:
                bboxes = self.detect_faces(image)
                # bboxes = self.face_detector.detect_faces(image=image,
                #                         conf_threshold=self.face_detection_threshold)
            except Exception as exc:
                raise exc
            # im = image.copy()
            # print(bboxes)
            # draw_bounding_box(im, bboxes[0])
            
            # draw_bounding_box(im, bboxes[1], color=(255,0,0))
            
            # cv2.imwrite('t1.jpg', im)
            # In order to register a person, we should make sure
            # only one face is present
            # if len(bboxes) != 1:
            #     raise NoFaceDetected if not len(bboxes) else MultipleFacesDetected
            # bbox = bboxes[0]

        # Convert the image back to RGB to feed to dlib based models
        # try:
        #     image = convert_to_rgb(image=image)
        # except InvalidImage:
        #     raise InvalidImage

        # print(bbox)
        encodings = []
        for bbox in bboxes:
            # draw_bounding_box(image, bbox, color=(255,0,0))
                
            # cv2.imwrite('data/faces/'+str(uuid.uuid4())+'.jpg', image)

            # Convert to dlib format rectangle
            # bbox = convert_to_dlib_rectangle(bbox)
            # Get the facial landmark coordinates
            face_keypoints = self.keypoints_detector(image, bbox)

            # Compute the 128D vector that describes the face in an img identified by
            # shape. In general, if two face descriptor vectors have a Euclidean
            # distance between them less than 0.6 then they are from the same
            # person, otherwise they are from different people. 
            face_encoding = self.get_face_encoding(image, face_keypoints)
            encodings.append(face_encoding)
        return encodings

    def get_facial_fingerprint(self, image, bbox=None):
        if bbox is None:
            raise FaceMissing
        
        # # If the input image is already a cropped ROI, no need to check 
        # # for number of faces in image
        # if check_face_count:
        #     # Face detection: If there are multiple faces detected
        #     # we dont register and throw an exception
        #     try:
        #         bboxes = self.face_detector.detect_faces(image=image,
        #                                 conf_threshold=self.face_detection_threshold)
        #     except Exception as exc:
        #         raise exc
        #     # im = image.copy()
        #     # print(bboxes)
        #     # draw_bounding_box(im, bboxes[0])
            
        #     # draw_bounding_box(im, bboxes[1], color=(255,0,0))
            
        #     # cv2.imwrite('t1.jpg', im)
        #     # In order to register a person, we should make sure
        #     # only one face is present
        #     # if len(bboxes) != 1:
        #     #     raise NoFaceDetected if not len(bboxes) else MultipleFacesDetected
        #     bbox = bboxes[0]

        # # Convert the image back to RGB to feed to dlib based models
        # try:
        #     image = convert_to_rgb(image=image)
        # except InvalidImage:
        #     raise InvalidImage

        # print(bbox)
        # draw_bounding_box(image, bbox, color=(255,0,0))
            
        # cv2.imwrite('data/faces/'+str(uuid.uuid4())+'.jpg', image)

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

    def get_face_encoding(self, image, face_keypoints):
        encoding = self.face_recognizor\
                    .compute_face_descriptor(image, face_keypoints, 1)
        return np.array(encoding)


    def euclidean_distance(self, vector1, vector2):
        return np.linalg.norm(vector1 - vector2) 


    


if __name__ == "__main__":
    from face_recog.media_utils import load_image_path

    ob = FaceRecognition(model_loc='models', 
                persistent_data_loc='data/facial_data.json',
                face_detector='mtcnn')
    img1 = load_image_path('data/sample/1.jpg')
    img2 = load_image_path('data/sample/2.jpg')
    img3 = load_image_path('data/sample/sagar.jpg')
    img4 = load_image_path('data/sample/vidit.jpg')
    img5 = load_image_path('data/sample/sagar2.jpg')
    # data1 = ob.register_face(image=img1, name='Test1')
    # data2 = ob.register_face(image=img2, name='Test2')
    
    # print(data1)
    # print(data2)

    # print('Match:', ob.euclidean_distance(data1['encoding'], data2['encoding']))
    
    ob.register_face(image=img1, name='Test1')
    ob.register_face(image=img2, name='Test2')
    ob.register_face(image=img4, name='Vidit')
    ob.register_face(image=img3, name='Sagar')

    fd = FaceDetectorMTCNN()
    fd2 = FaceDetectorOpenCV()
    print('FD',fd.detect_faces(img3))
    print('FD2',fd2.detect_faces(img3))
    
    print('Attempting face recognition...')
    match, dist = ob.recognize_face(img5, check_face_count=True)
    print(match['name'] if match and 'name' in match else '', dist)

    os.remove('data/facial_data.json')


    

