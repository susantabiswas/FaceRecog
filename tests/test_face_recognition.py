from face_recog.exceptions import InvalidImage, NoFaceDetected
from face_recog.face_recognition import FaceRecognition
import pytest
import numpy as np 
import cv2 
import dlib 

class TestFaceRecognition:
    def setup_class(self) -> None:
        self.face_recognizer = FaceRecognition(model_loc='models')

    def test_register_face(self, img2_data, img2_encoding):
        name = "Keanu"
        facial_data = self.face_recognizer.register_face(img2_data, name)
        # float comparison
        assert np.allclose(facial_data['encoding'], img2_encoding) \
                 == True


    def test_recognize_face(self, img2_data):
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


    def test_euclidean_distance(self):
        """ Check if euclidean distance computation works"""
        v1 = np.array((1, 2, 3)) 
        v2 = np.array((1, 1, 1)) 
        assert self.face_recognizer.euclidean_distance(v1, v2 ) == 2.23606797749979

    