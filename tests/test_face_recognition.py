from face_recog.validators import path_exists
from face_recog.exceptions import InvalidImage, NoFaceDetected
from face_recog.face_recognition import FaceRecognition
import pytest
import numpy as np 
import os
import math

class TestFaceRecognition:
    def setup_class(self) -> None:
        self.persistent_db_loc = 'data/test_facial_data.json'
        self.face_recognizer = FaceRecognition(model_loc='models')

    def teardown_method(self):
        if path_exists(self.persistent_db_loc):
            os.remove(self.persistent_db_loc)

    def test_register_face(self, img2_data, img2_encoding):
        name = "Test1"
        facial_data = self.face_recognizer.register_face(img2_data, name)
        # float comparison
        assert np.allclose(facial_data['encoding'], img2_encoding) \
                 == True


    def test_recognize_face(self, img1_data, img2_data):
        ob = FaceRecognition(model_loc='models', 
                persistent_data_loc=self.persistent_db_loc)
        
        ob.register_face(image=img1_data, name='Test1')
        match, dist = ob.recognize_face(img2_data)
        assert match['name'] == "Test1" and math.isclose(dist, 0.3834652785779021)

    def test_save_facial_data(self, face_data2, simple_cache_data2):
        ob = FaceRecognition(model_loc='models', 
                persistent_data_loc=self.persistent_db_loc)
        
        ob.save_facial_data(facial_data=face_data2)
        assert sorted(ob.get_registered_faces(),
                    key= lambda x: x["name"]) == \
                sorted([simple_cache_data2])

    def test_get_facial_fingerprint(self, img2_data, img2_encoding):
        facial_encoding = self.face_recognizer \
                            .get_facial_fingerprint(img2_data)
        # float comparison
        assert np.allclose(facial_encoding, img2_encoding) \
                 == True


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

    