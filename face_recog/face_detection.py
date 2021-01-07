# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
'''Description: Class for face detection. Uses a CNN 
based neural network to get the bounding box coordinates 
for a human face.

Usage: python -m face_recog.face_detection
'''
# ===================================================

from face_recog.exceptions import ModelFileMissing
import cv2
import os  
from typing import List

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
            self.face_detector = cv2.dnn.readNetFromTensorflow(model_path,
                                                        config_path)
        except Exception as e:
            raise e


    def model_inference(self, image)->List:
        # Run the face detection model on the image to get 
        # bounding box coordinates
        # The model expects input as a blob, create input image blob
        img_blob = cv2.dnn.blobFromImage(image, 1.0, \
                        (300, 300), [104, 117, 123], False, False)
        # Feed the input blob to NN and get the output layer predictions
        self.face_detector.setInput(img_blob)
        detections = self.face_detector.forward()

        return detections


    def detect_face(self, image, 
                    conf_threshold: float=0.7)->List[List[int]]:
        if image is None:
            return []

        # To prevent modification of orig img
        image = image.copy()
        height, width = image.shape[:2]
        
        # Do a forward propagation with the blob created from input img
        detections = self.model_inference(image)
        # Bounding box coordinates of faces in image
        bboxes = []
        for idx in range(detections.shape[2]):
            conf = detections[0, 0, idx, 2]
            if conf >= conf_threshold:
                # Scale the bbox coordinates to suit image
                x1 = int(detections[0, 0, idx, 3] * width)
                y1 = int(detections[0, 0, idx, 4] * height)
                x2 = int(detections[0, 0, idx, 5] * width)
                y2 = int(detections[0, 0, idx, 6] * height)

                bboxes.append([x1, y1, x2, y2])

        return bboxes


    def __repr__(self):
        return "FaceDetector"

if __name__ == "__main__":
    # Sample Usage
    ob = FaceDetector(model_loc='models')
    img = cv2.imread('data/sample/2.jpg')
    bbox = ob.detect_face(img)
    print(bbox)
    print(ob)
