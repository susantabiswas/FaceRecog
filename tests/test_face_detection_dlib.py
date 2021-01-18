import pytest
from face_recog.exceptions import ModelFileMissing
from face_recog.face_detection_dlib import FaceDetectorDlib


def test_correct_model_path():
    """
    Test object init with the correct model path
    """
    ob = None
    model_loc = "./models"
    try:
        ob = FaceDetectorDlib(model_loc=model_loc, model_type="mmod")
    except Exception:
        pass
    finally:
        assert isinstance(ob, FaceDetectorDlib)


def test_incorrect_model_path():
    """
    Test object init with the incorrect model path
    """
    ob = None
    incorrect_model_loc = "./wrong_models"
    with pytest.raises(ModelFileMissing):
        ob = FaceDetectorDlib(model_loc=incorrect_model_loc, model_type="mmod")


def test_detect_face_hog(img2_data, img2_facebox_dlib_hog):
    model_loc = "./models"
    ob = FaceDetectorDlib(model_loc=model_loc, model_type="hog")
    assert img2_facebox_dlib_hog == ob.detect_faces(img2_data)


def test_detect_face_mmod(img2_data, img2_facebox_dlib_mmod):
    model_loc = "./models"
    ob = FaceDetectorDlib(model_loc=model_loc, model_type="mmod")
    assert img2_facebox_dlib_mmod == ob.detect_faces(img2_data)
