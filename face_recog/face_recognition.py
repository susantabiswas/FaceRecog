# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
"""Description: Class for Face Recognition related methods.
Main operations: Register and Recognize face.

Usage: python -m face_recog.face_recognition

dlib model files can be downloaded from:
http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2\n"
http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"

https://github.com/davisking/dlib-models
"""
# ===================================================

import os
import sys
import uuid
from typing import Dict, List, Tuple

import dlib
import numpy as np

from face_recog.exceptions import (
    FaceMissing,
    InvalidImage,
    ModelFileMissing,
    NoFaceDetected,
    NoNameProvided,
)
from face_recog.face_data_store import FaceDataStore
from face_recog.face_detection_dlib import FaceDetectorDlib
from face_recog.face_detection_mtcnn import FaceDetectorMTCNN
from face_recog.face_detection_opencv import FaceDetectorOpenCV
from face_recog.logger import LoggerFactory
from face_recog.media_utils import convert_to_dlib_rectangle
from face_recog.validators import is_valid_img, path_exists

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


class FaceRecognition:
    """Class for Face Recognition related methods.
    Main operations: Register and Recognize face.

    Raises:
        ModelFileMissing: [description]
        NoNameProvided: [description]
        NoFaceDetected: [description]
        FaceMissing: [description]
    """

    keypoints_model_path = "shape_predictor_5_face_landmarks.dat"
    face_recog_model_path = "dlib_face_recognition_resnet_model_v1.dat"

    def __init__(
        self,
        model_loc: str = "./models",
        persistent_data_loc="data/facial_data.json",
        face_detection_threshold: int = 0.99,
        face_detector: str = "dlib",
    ) -> None:
        """Constructor

        Args:
            model_loc (str, optional): Path where model files are saved. Defaults to "./models".
            persistent_data_loc (str, optional): Path to save the persistence storage file.
                Defaults to 'data/facial_data.json'.
            face_detection_threshold (int, optional): Threshold facial model confidence to consider a detection.
                Defaults to 0.99.
            face_detector (str, optional): Type of face detector to use. Options:
                Dlib-HOG and MMOD, MTCNN, OpenCV CNN. Defaults to 'dlib'.

        Raises:
            ModelFileMissing: Raised when model file is not found
        """
        keypoints_model_path = os.path.join(
            model_loc, FaceRecognition.keypoints_model_path
        )
        face_recog_model_path = os.path.join(
            model_loc, FaceRecognition.face_recog_model_path
        )
        if not (
            path_exists(keypoints_model_path) or path_exists(face_recog_model_path)
        ):
            raise ModelFileMissing
        if face_detector == "opencv":
            self.face_detector = FaceDetectorOpenCV(
                model_loc=model_loc, crop_forehead=True, shrink_ratio=0.2
            )
        elif face_detector == "mtcnn":
            self.face_detector = FaceDetectorMTCNN(crop_forehead=True, shrink_ratio=0.2)
        else:
            self.face_detector = FaceDetectorDlib()
        self.face_detection_threshold = face_detection_threshold

        self.keypoints_detector = dlib.shape_predictor(keypoints_model_path)
        self.face_recognizor = dlib.face_recognition_model_v1(face_recog_model_path)
        self.datastore = FaceDataStore(persistent_data_loc=persistent_data_loc)

    def register_face(self, image=None, name: str = None, bbox: List[int] = None):
        """Method to register a face via the facial encoding.
        Siamese neural network is used to generate 128 numbers
        for a given facial region. These encodings can be used to identify a
        facial ROI for identification later.

        Args:
            image (numpy array, optional): Defaults to None.
            name (str, optional): Name to associate with the face. Defaults to None.
            bbox (List[int], optional): Facial ROI bounding box. Defaults to None.

        Raises:
            NoNameProvided:
            NoFaceDetected:

        Returns:
            Dict: Facial encodings along with an unique identifier and name
        """

        if not is_valid_img(image) or name is None:
            raise NoNameProvided if name is None else InvalidImage

        image = image.copy()
        face_encoding = None

        try:
            if bbox is None:
                bboxes = self.face_detector.detect_faces(image=image)
                if len(bboxes) == 0:
                    raise NoFaceDetected
                bbox = bboxes[0]
            face_encoding = self.get_facial_fingerprint(image, bbox)

            # Convert the numpy array to normal python float list
            # to make json serialization simpler
            facial_data = {
                "id": str(uuid.uuid4()),
                "encoding": tuple(face_encoding.tolist()),
                "name": name,
            }
            # save the encoding with the name
            self.save_facial_data(facial_data)
            logger.info("Face registered with name: {}".format(name))
        except Exception as exc:
            raise exc
        return facial_data

    def save_facial_data(self, facial_data: Dict = None) -> bool:
        """Saves facial data to cache and persistent storage

        Args:
            facial_data (Dict, optional): [description]. Defaults to None.

        Returns:
            bool: status of saving
        """
        if facial_data is not None:
            self.datastore.add_facial_data(facial_data=facial_data)
            return True
        return False

    def get_registered_faces(self) -> List[Dict]:
        """Returns the list of all facial data of all registered users

        Returns:
            List[Dict]: List of facial data
        """
        return self.datastore.get_all_facial_data()

    def recognize_faces(
        self, image, threshold: float = 0.6, bboxes: List[List[int]] = None
    ):
        """Finds matching registered users for the
        face(s) in the input image. The input image should be cropped to contain
        only one face and then sent to this method.

        Args:
            image (numpy array): [description]
            threshold (float, optional): Max threshold euclidean distance to
            consider two people to be a match. Defaults to 0.6.
            bboxes (List[List[int]], optional): List of facial ROI bounding box.
                If this is None, then face detection is performed on the image
                and facial recognition is run for all the detected faces, otherwise
                if a bounding box is sent, then facial recognition is only
                done for that bounding box. Defaults to None.

        Raises:
            NoFaceDetected: [description]

        Returns:
            List[Tuple]: List of information of matching
        """
        if image is None:
            return InvalidImage
        image = image.copy()

        if bboxes is None:
            bboxes = self.face_detector.detect_faces(image=image)
            if len(bboxes) == 0:
                raise NoFaceDetected
        # Load the data of existing registered faces
        # compare using the metric the closest match
        all_facial_data = self.datastore.get_all_facial_data()
        matches = []
        for bbox in bboxes:
            face_encoding = self.get_facial_fingerprint(image, bbox)
            match, min_dist = None, 10000000

            for face_data in all_facial_data:
                dist = self.euclidean_distance(face_encoding, face_data["encoding"])
                if dist <= threshold and dist < min_dist:
                    match = face_data
                    min_dist = dist
            # bound box, matched face details, dist from closest match
            matches.append((bbox, match, min_dist))
        return matches

    def get_facial_fingerprint(self, image, bbox: List[int] = None) -> List[float]:
        """Driver method for generating the facial encoding for an input image.
            Input image bbox -> facial keypoints detection -> keypoints used for
            face alignment -> Siamese NN -> Encoding
        Args:
            image (numpy array): [description]
            bbox (List[int], optional): List of facial ROI bounding box. Defaults to None.

        Raises:
            FaceMissing: [description]

        Returns:
            List[float]: Facial Encoding
        """
        if bbox is None:
            raise FaceMissing
        # Convert to dlib format rectangle
        bbox = convert_to_dlib_rectangle(bbox)
        # Get the facial landmark coordinates
        face_keypoints = self.keypoints_detector(image, bbox)

        # Compute the 128D vector that describes the face in an img identified by
        # shape. In general, if two face descriptor vectors have a Euclidean
        # distance between them less than 0.6 then they are from the same
        # person, otherwise they are from different people.
        face_encoding = self.get_face_encoding(image, face_keypoints)
        return face_encoding

    def get_face_encoding(self, image, face_keypoints: List):
        """Method for generating the facial encoding for
            a face in an input image.

        Args:
            image (numpy array): [description]
            face_keypoints (List): [description]

        Returns:
            [type]: [description]
        """
        encoding = self.face_recognizor.compute_face_descriptor(
            image, face_keypoints, 1
        )
        return np.array(encoding)

    def euclidean_distance(self, vector1: Tuple, vector2: Tuple):
        """Computes Euclidean distance between two vectors

        Args:
            vector1 (Tuple): [description]
            vector2 (Tuple): [description]

        Returns:
            [type]: [description]
        """
        return np.linalg.norm(np.array(vector1) - np.array(vector2))


if __name__ == "__main__":
    ############ Sample Usage and Testing ################
    # from face_recog.media_utils import load_image_path

    # ob = FaceRecognition(
    #     model_loc="models",
    #     persistent_data_loc="data/facial_data.json",
    #     face_detector="dlib",
    # )
    # img1 = load_image_path("data/sample/1.jpg")
    # img2 = load_image_path("data/sample/2.jpg")
    # img3 = load_image_path("data/sample/sagar.jpg")
    # img4 = load_image_path("data/sample/vidit.jpg")
    # img5 = load_image_path("data/sample/sagar2.jpg")
    
    # data1 = ob.register_face(image=img1, name='Keanu')
    # data2 = ob.register_face(image=img2, name='Test2')

    # # print(data1)
    # print(data2)

    # fd = FaceDetectorMTCNN()
    # fd2 = FaceDetectorOpenCV()
    # print('FD',fd.detect_faces(img3))
    # print('FD2',fd2.detect_faces(img3))

    # print('Attempting face recognition...')
    # matches = ob.recognize_faces(img5)
    # print(match['name'] if match and 'name' in match else '', dist)

    # os.remove("data/facial_data.json")
    pass
