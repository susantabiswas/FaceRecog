# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
'''Description: Class for Face Recognition related methods.
Main operations: Register and Recognize face.

Usage: python -m face_recog.face_recognition
'''
# ===================================================

from face_recog.validators import is_valid_img
import cv2
from face_recog.exceptions import (InvalidImage, ModelFileMissing, 
            NoFaceDetected, MultipleFacesDetected)
from face_recog.face_detection import FaceDetector
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

        self.face_detector = FaceDetector(model_loc=model_loc)
        self.keypoints_detector = dlib.shape_predictor(keypoints_model_path)
        self.face_recognizor = dlib.face_recognition_model_v1(face_recog_model_path)


    def register_face(self, image, name):
        """ Registers a single face"""
        image = image.copy()

        # Face detection: If there are multiple faces detected
        # we dont register and throw an exception
        try:
            bboxes = self.face_detector.detect_face(image=image)
        except Exception as exc:
            raise exc

        # In order to register a person, we should make sure
        # only one face is present
        if len(bboxes) != 1:
            raise NoFaceDetected if not len(bboxes) else MultipleFacesDetected
        # Convert the image back to RGB to feed to dlib based models
        try:
            image = self.convert_to_rgb(image=image)
        except InvalidImage:
            raise InvalidImage

        # Convert to dlib format rectangle
        bbox = self.convert_to_dlib_rectangle(bboxes[0])
        # Get the facial landmark coordinates
        face_keypoints = self.keypoints_detector(image, bbox)

        # Compute the 128D vector that describes the face in an img identified by
        # shape. In general, if two face descriptor vectors have a Euclidean
        # distance between them less than 0.6 then they are from the same
        # person, otherwise they are from different people. 
        face_encoding = self.get_face_encoding(image, face_keypoints)
        
        facial_data = {
            "id": uuid.uuid4(),
            "encoding": face_encoding,
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


    def convert_to_dlib_rectangle(self, bbox):
        return dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])

    
    def load_image_path(self, img_path):
        try:
            img = cv2.imread(img_path)
            return img
        except Exception as exc:
            raise exc

    def convert_to_rgb(self, image):
        if not is_valid_img(image):
            raise InvalidImage
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# https://github.com/davisking/dlib-models
# import sys
# import os
# import dlib
# import glob

# # if len(sys.argv) != 4:
# #     print(
# #         "Call this program like this:\n"
# #         "   ./face_recognition.py shape_predictor_5_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ../examples/faces\n"
# #         "You can download a trained facial shape predictor and recognition model from:\n"
# #         "    http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
# #         "    http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
# #     exit()

# predictor_path = 'models/shape_predictor_5_face_landmarks.dat'
# face_rec_model_path = 'models/dlib_face_recognition_resnet_model_v1.dat'

# #faces_folder_path = sys.argv[3]

# # Load all the models we need: a detector to find the faces, a shape predictor
# # to find face landmarks so we can precisely localize the face, and finally the
# # face recognition model.
# detector = dlib.get_frontal_face_detector()
# sp = dlib.shape_predictor(predictor_path)
# facerec = dlib.face_recognition_model_v1(face_rec_model_path)

# win = dlib.image_window()

# img = dlib.load_rgb_image('data/sample/2.jpg')

# ob = FaceDetector(model_loc='models')
# # img = cv2.imread('data/sample/2.jpg')
# bbox = ob.detect_face(img)
# print(bbox)
# dets = [dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3]) for bbox in bbox]

# # win.clear_overlay()
# # win.set_image(img)

# # Ask the detector to find the bounding boxes of each face. The 1 in the
# # second argument indicates that we should upsample the image 1 time. This
# # will make everything bigger and allow us to detect more faces.
# # dets = detector(img, 1)
# print('***',dets)
# print("Number of faces detected: {}".format(len(dets)))

# face_descriptor = None
# # Now process each face we found.
# for k, d in enumerate(dets):
#     print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#         k, d.left(), d.top(), d.right(), d.bottom()))
#     # Get the landmarks/parts for the face in box d.
#     shape = sp(img, d)
#     # Draw the face landmarks on the screen so we can see what face is currently being processed.
#     # win.clear_overlay()
#     # win.add_overlay(d)
#     # win.add_overlay(shape)

#     # Compute the 128D vector that describes the face in img identified by
#     # shape.  In general, if two face descriptor vectors have a Euclidean
#     # distance between them less than 0.6 then they are from the same
#     # person, otherwise they are from different people. Here we just print
#     # the vector to the screen.
#     face_descriptor = facerec.compute_face_descriptor(img, shape)
#     # print(face_descriptor)
#     # It should also be noted that you can also call this function like this:
#     #  face_descriptor = facerec.compute_face_descriptor(img, shape, 100, 0.25)
#     # The version of the call without the 100 gets 99.13% accuracy on LFW
#     # while the version with 100 gets 99.38%.  However, the 100 makes the
#     # call 100x slower to execute, so choose whatever version you like.  To
#     # explain a little, the 3rd argument tells the code how many times to
#     # jitter/resample the image.  When you set it to 100 it executes the
#     # face descriptor extraction 100 times on slightly modified versions of
#     # the face and returns the average result.  You could also pick a more
#     # middle value, such as 10, which is only 10x slower but still gets an
#     # LFW accuracy of 99.3%.
#     # 4th value (0.25) is padding around the face. If padding == 0 then the chip will
#     # be closely cropped around the face. Setting larger padding values will result a looser cropping.
#     # In particular, a padding of 0.5 would double the width of the cropped area, a value of 1.
#     # would triple it, and so forth.

#     # There is another overload of compute_face_descriptor that can take
#     # as an input an aligned image. 
#     #
#     # Note that it is important to generate the aligned image as
#     # dlib.get_face_chip would do it i.e. the size must be 150x150, 
#     # centered and scaled.
#     #
#     # Here is a sample usage of that

#     print("Computing descriptor on aligned image ..")
    
#     # Let's generate the aligned image using get_face_chip
#     face_chip = dlib.get_face_chip(img, shape)        

#     # Now we simply pass this chip (aligned image) to the api
#     face_descriptor_from_prealigned_image = facerec.compute_face_descriptor(face_chip)                
#     # print(face_descriptor_from_prealigned_image)        
    
#     # dlib.hit_enter_to_continue()


# ob1 = FaceRecognition()
# image = dlib.load_rgb_image('data/sample/2.jpg')
# bbox = ob.detect_face(image)[0]
# e2 = ob1.register_face(image, bbox)

# print(ob1.euclidean_distance(e2['encoding'], face_descriptor))


if __name__ == "__main__":
    ob = FaceRecognition(model_loc='models')
    img1 = ob.load_image_path('data/sample/1.jpg')
    img2 = ob.load_image_path('data/sample/2.jpg')

    print(img1.shape)
    data1 = ob.register_face(image=img1, name='Test1')
    # data2 = ob.register_face(image=img2, name='Test2')

    print(data1)
    # print(data2)

    # print('Match:', ob.euclidean_distance(data1['encoding'], data2['encoding']))

