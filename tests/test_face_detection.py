from exceptions import ModelFileMissing
from face_detection import FaceDetector
import unittest
import cv2
import os

class TestFaceDetection(unittest.TestCase):
    def __init__(self, img_loc='data/sample/1.jpg') -> None:
        self.img = cv2.imread(img_loc)

    def test_correct_model_path(self, model_loc='./models'):
        """
            Test object init with the correct model path 
        """
        ob = None
        try:
            ob = FaceDetector(model_loc=model_loc)
        except Exception: 
            pass
        finally:
            self.assertIsInstance(ob, FaceDetector)

    def test_incorrect_model_path(self, 
            inccorrect_model_loc='./wrong_models'):
        """
            Test object init with the incorrect model path 
        """
        ob = None
        with self.assertRaises(ModelFileMissing):
            ob = FaceDetector(model_loc=inccorrect_model_loc)
       


if __name__ == "__main__":
    unittest.main()