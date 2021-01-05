import pytest
from face_recog.exceptions import ModelFileMissing
from face_recog.face_detection import FaceDetector

def test_correct_model_path(benchmark):
        """
            Test object init with the correct model path 
        """
        ob = None
        model_loc='./models'
        try:
            ob = benchmark(FaceDetector(model_loc=model_loc))
        except Exception: 
            pass
        finally:
            print('*****************************',ob)
            assert isinstance(ob, FaceDetector)

def test_incorrect_model_path():
    """
        Test object init with the incorrect model path 
    """
    ob = None
    inccorrect_model_loc='./wrong_models'
    with pytest.raises(ModelFileMissing):
        ob = FaceDetector(model_loc=inccorrect_model_loc)


test_correct_model_path()