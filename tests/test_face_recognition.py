from face_recog.exceptions import InvalidImage, NoFaceDetected
from face_recog.face_recognition import FaceRecognition
import pytest
import numpy as np 
import cv2 
import dlib 

class TestFaceRecognition:
    def setup(self) -> None:
        self.face_recognizer = FaceRecognition(model_loc='models')

    def test_register_face(self, img1_data):
        pass

    def test_recognize_face(self, img1_data):
        pass

    def test_save_facial_data(self):
        facial_data = []
        pass


    def test_recognize_face_missing_face(self):
        """ Check if exception is thrown when no face is visible"""
        # create a dummy image
        img = np.zeros((100, 100, 3), dtype='float32')
        with pytest.raises(NoFaceDetected):
            self.face_recognizer.register_face(image=img, name="test1")


    def test_recognize_face_invalid_image(self):
        """ Check if exception is thrown when an invalid array is given"""
        # create a dummy image
        img = np.zeros((100, 100, 5), dtype='float32')
        with pytest.raises(InvalidImage):
            self.face_recognizer.register_face(img, "test1")


    def test_convert_to_dlib_rectangle(self):
        """ Check if dlib rectangle is created properly"""
        bbox = [1, 2, 3, 4]
        dlib_box = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
        assert self.face_recognizer.convert_to_dlib_rectangle(bbox) == dlib_box
    

    def test_load_image_path(self):
        """ Check if exception is thrown when an invalid array is given"""
        path = 'data/sample/1.jpg'
        img = cv2.imread(path)
        loaded_img = self.face_recognizer.load_image_path(path)
        assert np.all(loaded_img == img) == True


    def test_convert_to_rgb_exception(self):
        """ Check if exception is thrown when an invalid array is given"""
        # create a dummy image
        img = np.zeros((100, 100, 5))
        with pytest.raises(InvalidImage):
            self.face_recognizer.convert_to_rgb(img)


    def test_convert_to_rgb(self, img1_data):
        """ Check if RGB conversion happens correctly"""
        rgb = cv2.cvtColor(img1_data, cv2.COLOR_BGR2RGB)
        converted_img = self.face_recognizer.convert_to_rgb(img1_data)
        assert np.all(rgb == converted_img) == True

    def test_euclidean_distance(self):
        """ Check if euclidean distance computation works"""
        v1 = np.array((1, 2, 3)) 
        v2 = np.array((1, 1, 1)) 
        assert self.face_recognizer.euclidean_distance(v1, v2 ) == 2.23606797749979

    