import cv2
from face_recog.validators import is_valid_img
from face_recog.exceptions import InvalidImage
import dlib 

def convert_to_rgb(image):
    if not is_valid_img(image):
        raise InvalidImage
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_to_dlib_rectangle(bbox):
    """Converts a bounding box coordinate list 
    to dlib rectangle.

    Args:
        bbox (List[int]): Bounding box coordinates

    Returns:
        dlib.rectangle: Dlib rectangle
    """
    return dlib.rectangle(bbox[0], bbox[1],
                         bbox[2], bbox[3])


def load_image_path(img_path):
    try:
        img = cv2.imread(img_path)
        return img
    except Exception as exc:
        raise exc

def draw_bounding_box(image, bbox, color=(0,255,0)):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2),
                        color, 2)
    return image

def draw_annotation(image, name, bbox, color=(0, 255, 0)):
    draw_bounding_box(image, bbox, color=color)
    x1, y1, x2, y2 = bbox
    
    # Draw the label with name below the face
    cv2.rectangle(image, (x1, y2 - 20), (x2, y2), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (x1 + 6, y2 - 6), font, 0.6, (0, 0, 0), 2)
    

def get_facial_ROI(image, bbox):
    if image is None or bbox is None:
        raise InvalidImage if image is None else ValueError
    return image[bbox[1]:bbox[3],
                bbox[0]: bbox[2], :]

def get_video_writer(video_stream,
                    output_filename:str):
    """Returns an OpenCV video writer with mp4 codec stream

    Args:
        video_stream (OpenCV video stream obj): Input video stream
        output_filename (str):

    Returns:
        OpenCV VideoWriter: 
    """
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        FPS = video_stream.get(cv2.CAP_PROP_FPS)
        
        # (Width, Height)
        dims = (int(video_stream.get(3)), 
                int(video_stream.get(4)))
        video_writer = cv2.VideoWriter(
                                output_filename, 
                                fourcc, FPS, 
                                dims
                            )

        return video_writer

    except Exception as exc:
        raise exc