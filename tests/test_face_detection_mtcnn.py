import pytest
from face_recog.exceptions import ModelFileMissing
from face_recog.face_detection_mtcnn import FaceDetectorMTCNN


def test_detect_face(img2_data, img2_facebox_mtcnn):
    ob = FaceDetectorMTCNN()
    assert img2_facebox_mtcnn == ob.detect_faces(img2_data)
