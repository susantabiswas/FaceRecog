import pytest
from face_recog.exceptions import ModelFileMissing
from face_recog.face_detection import FaceDetector
import cv2 

def test_correct_model_path():
        """
            Test object init with the correct model path 
        """
        ob = None
        model_loc='./models'
        try:
            ob = FaceDetector(model_loc=model_loc)
        except Exception: 
            pass
        finally:
            assert isinstance(ob, FaceDetector)

def test_incorrect_model_path():
    """
        Test object init with the incorrect model path 
    """
    ob = None
    inccorrect_model_loc='./wrong_models'
    with pytest.raises(ModelFileMissing):
        ob = FaceDetector(model_loc=inccorrect_model_loc)


def test_detect_face(img_data):
    model_loc='./models'
    ob = FaceDetector(model_loc=model_loc)
    assert [[348, 76, 407, 166]] == ob.detect_face(img_data)