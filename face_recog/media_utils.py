import cv2
from face_recog.validators import is_valid_img
from face_recog.exceptions import InvalidImage
import dlib 

def convert_to_rgb(image):
    if not is_valid_img(image):
        raise InvalidImage
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_to_dlib_rectangle(bbox):
    return dlib.rectangle(bbox[0], bbox[1],
                         bbox[2], bbox[3])


def load_image_path(img_path):
    try:
        img = cv2.imread(img_path)
        return img
    except Exception as exc:
        raise exc