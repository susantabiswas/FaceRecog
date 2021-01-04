from exceptions import ModelFileMissing
import cv2
import os  

class FaceDetector:
    def __init__(self, model_loc='./models'):
        # Model file and associated config path
        model_path = os.path.join(model_loc,
                            'opencv_face_detector_uint8.pb')
        config_path = os.path.join(model_loc,
                            'opencv_face_detector.pbtxt')

        if not os.path.exists(model_path) or \
            not os.path.exists(config_path):
            raise ModelFileMissing
        try:
            # load the model
            self.face_detector = cv2.dnn.readFromTensorflow(model_path,
                                                        config_path)
        except Exception as e:
            raise e

    def detect_face(self, image):
        pass
