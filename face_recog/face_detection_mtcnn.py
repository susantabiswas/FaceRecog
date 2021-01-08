# You can install mtcnn using PIP by typing "pip install mtcnn"
from mtcnn import MTCNN
import cv2
 
detector = MTCNN()

img = cv2.cvtColor(cv2.imread("data/sample/1.jpg"), cv2.COLOR_BGR2RGB)
 
print(detector.detect_faces(img))
