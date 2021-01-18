import pytest
from face_recog.exceptions import ModelFileMissing
from face_recog.face_detection_opencv import FaceDetectorOpenCV


def test_correct_model_path():
    """
    Test object init with the correct model path
    """
    ob = None
    model_loc = "./models"
    try:
        ob = FaceDetectorOpenCV(model_loc=model_loc)
    except Exception:
        pass
    finally:
        assert isinstance(ob, FaceDetectorOpenCV)


def test_incorrect_model_path():
    """
    Test object init with the incorrect model path
    """
    ob = None
    inccorrect_model_loc = "./wrong_models"
    with pytest.raises(ModelFileMissing):
        _ = FaceDetectorOpenCV(model_loc=inccorrect_model_loc)


def test_detect_face(img2_data, img2_facebox_opencv):
    model_loc = "./models"
    ob = FaceDetectorOpenCV(model_loc=model_loc)
    assert img2_facebox_opencv == ob.detect_faces(img2_data)
