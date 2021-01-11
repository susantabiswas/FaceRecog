# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
'''Description: Class with methods to do facial recognition
on video or webcam feed.

Usage: python -m face_recog.video_main'''
# ===================================================
import traceback
from face_recog.exceptions import NoNameProvided
from face_recog.media_utils import draw_annotation, draw_bounding_box, get_facial_ROI
from face_recog.face_detection_opencv import FaceDetectorOpenCV
from face_recog.face_recognition import FaceRecognition
from face_recog.face_detection_mtcnn import FaceDetectorMTCNN
import cv2
import sys

class FaceRecognitionVideo:
    def __init__(self, face_detector='opencv', 
                model_loc='models', 
                persistent_db_path='data/facial_data.json',
                face_detection_threshold=0.8) -> None:
        self.face_recognizer = FaceRecognition(model_loc=model_loc,
                                        persistent_data_loc=persistent_db_path,
                                        face_detection_threshold=face_detection_threshold)
        self.face_detector = FaceDetectorOpenCV(model_loc=model_loc) \
                                if face_detector == "opencv" \
                            else FaceDetectorMTCNN()

    def recognize_face_video(self, video_path=None,
                detection_interval:int=15):
        if video_path is None:
            # If no video source is given, try
            # switching to webcam
            video_path = 0    

        try:
            cap = cv2.VideoCapture(video_path)
            # height, width = cap.get(4), cap.get(3)
            frame_num = 0
            match, bboxes = None, None
                
            while True:
                status, frame = cap.read()
                if not status:
                    break
                
                if frame_num % detection_interval == 0:
                    # detect faces
                    bboxes = self.face_detector.detect_faces(image=frame)
                    
                for bbox in bboxes:
                    # crop the facial region
                    face = get_facial_ROI(frame, bbox)
                    if frame_num % detection_interval == 0:           
                        match, _ = self.face_recognizer\
                                    .recognize_face(image = face, threshold=0.6,
                                                check_face_count=False)
                    name = match['name'] if match is not None else ''
                    # draw boudning box
                    draw_annotation(frame, name, bbox)
                    # draw face labels
                    print('Match: {}'.format(name))
                
                cv2.imshow('Preview', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                frame_num += 1
                
        except Exception as exc:
            raise exc
        finally:
            cv2.destroyAllWindows()

    
    def register_face_webcam(self, name=None, detection_interval=5):
        if name is None:
            raise NoNameProvided
        try:
            cap = cv2.VideoCapture(0)
            frame_num = 0
            registered_face = None

            while True:
                status, frame = cap.read()
                if not status:
                    break
                
                if frame_num % detection_interval == 0:
                    # detect faces
                    bboxes = self.face_detector.detect_faces(image=frame)
                    try:
                        if len(bboxes) == 1: 
                            face = get_facial_ROI(frame, bboxes[0])
                            facial_data = self.face_recognizer \
                                                .register_face(
                                                    image=face, 
                                                    name=name,
                                                    check_face_count=False)
                            if facial_data:
                                registered_face = face
                                break
                    except Exception as exc:
                        traceback.print_exc(file=sys.stdout)
                        
                frame_num += 1
            cv2.imshow('Registered Face', registered_face)
            print('[INFO]Press any key to continue......')
            cv2.waitKey(0)
                                    
        except Exception as exc:
            raise exc
        finally:
            cv2.destroyAllWindows()

            
ob = FaceRecognitionVideo(face_detector='opencv')
# ob.recognize_face_video(video_path=None,
#                         detection_interval=5)
# register a face
ob.register_face_webcam(name="susanta")