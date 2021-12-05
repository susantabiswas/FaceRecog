# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Class for face detection. Uses a MTCNN 
based neural network to get the bounding box coordinates 
for a human face.

Usage: python -m face_recog.face_detection_mtcnn

You can install mtcnn using PIP by typing "pip install mtcnn"
Ref: https://github.com/ipazc/mtcnn
"""
# ===================================================

import sys
from typing import List

from mtcnn import MTCNN

from face_recog.exceptions import InvalidImage
from face_recog.face_detector import FaceDetector
from face_recog.logger import LoggerFactory
from face_recog.validators import is_valid_img

# Load the custom logger
logger = None
try:
    logger_ob = LoggerFactory(logger_name=__name__)
    logger = logger_ob.get_logger()
    logger.info("{} loaded...".format(__name__))
    # set exception hook for uncaught exceptions
    sys.excepthook = logger_ob.uncaught_exception_hook
except Exception as exc:
    raise exc


class FaceDetectorMTCNN(FaceDetector):
    """Class for face detection. Uses a MTCNN
    based neural network to get the bounding box coordinates
    for a human face.
    """

    def __init__(self, crop_forehead: bool = True, shrink_ratio: int = 0.1):
        """Constructor

        Args:
            crop_forehead (bool, optional): Whether to trim the
                forehead in the detected facial ROI. Certain datasets
                like Dlib models are trained on cropped images without forehead.
                It can useful in those scenarios.
                Defaults to True.
            shrink_ratio (float, optional): Amount of height to shrink
                Defaults to 0.1
        """
        try:
            # load the model
            self.face_detector = MTCNN()
            self.crop_forehead = crop_forehead
            self.shrink_ratio = shrink_ratio
            logger.info("MTCNN face detector loaded...")
        except Exception as e:
            raise e

    def detect_faces(self, image, conf_threshold: float = 0.7) -> List[List[int]]:
        """Performs facial detection on an image. Uses MTCNN.
        Args:
            image (numpy array):
            conf_threshold (float, optional): Threshold confidence to consider
        Raises:
            InvalidImage: When the image is either None or
            with wrong number of channels.

        Returns:
            List[List[int]]: List of bounding box coordinates
        """
        if not is_valid_img(image):
            raise InvalidImage

        # Do a forward propagation with the blob created from input img
        detections = self.face_detector.detect_faces(image)
        # Bounding box coordinates of faces in image
        bboxes = []
        for _, detection in enumerate(detections):
            conf = detection["confidence"]
            if conf >= conf_threshold:
                x, y, w, h = detection["box"]
                x1, y1, x2, y2 = x, y, x + w, y + h
                # Trim forehead area to match dlib style facial ROI
                if self.crop_forehead:
                    y1 = y1 + int(h * self.shrink_ratio)
                bboxes.append([x1, y1, x2, y2])

        return bboxes


    def __repr__(self):
        return "FaceDetectorMTCNN"


if __name__ == "__main__":

    ############ Sample Usage
    # ob = FaceDetectorMTCNN(crop_forehead=False)
    # img = cv2.imread("data/sample/1.jpg")

    # # import numpy as np
    # # img = np.zeros((100,100,5), dtype='float32')
    # bbox = ob.detect_faces(convert_to_rgb(img))
    # cv2.imwrite('data/out1.jpg', img)
    pass
