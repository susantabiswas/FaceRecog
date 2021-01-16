# ---- coding: utf-8 ----
# ===================================================
# Author: Susanta Biswas
# ===================================================
'''Description: Class with methods to do facial recognition
on video or webcam feed.

Usage: python -m face_recog.video_main'''
# ===================================================
from face_recog.face_detection_dlib import FaceDetectorDlib
from face_recog.validators import path_exists
import traceback
from face_recog.exceptions import NoNameProvided, PathNotFound
from face_recog.media_utils import convert_to_rgb, draw_annotation, draw_bounding_box, get_facial_ROI, get_video_writer
from face_recog.face_detection_opencv import FaceDetectorOpenCV
from face_recog.face_recognition import FaceRecognition
from face_recog.face_detection_mtcnn import FaceDetectorMTCNN
import cv2
import sys
import time
import numpy as np
                            

class FaceRecognitionVideo:
    def __init__(self, face_detector='dlib', 
                model_loc='models', 
                persistent_db_path='data/facial_data.json',
                face_detection_threshold=0.8) -> None:

        self.face_recognizer = FaceRecognition(model_loc=model_loc,
                                        persistent_data_loc=persistent_db_path,
                                        face_detection_threshold=face_detection_threshold,
                                        face_detector=face_detector)
        if face_detector == "opencv":
            self.face_detector = FaceDetectorOpenCV(model_loc=model_loc,
                                                    crop_forehead=True, 
                                                    shrink_ratio=0.2) 
        elif face_detector == "mtcnn":
            self.face_detector = FaceDetectorMTCNN(crop_forehead=True, 
                                                    shrink_ratio=0.2)
        elif face_detector == "dlib":
            self.face_detector = FaceDetectorDlib()

    def recognize_face_video(self, video_path=None,
                detection_interval:int=15, 
                save_output:bool=False,
                preview:bool=False,
                output_path='data/output.mp4',
                verbose:bool=True):

        if video_path is None:
            # If no video source is given, try
            # switching to webcam
            video_path = 0    
        cap, video_writer = None, None

        try:
            cap = cv2.VideoCapture(video_path)
            # height, width = cap.get(4), cap.get(3)

            video_writer = get_video_writer(cap, output_path)
    
            frame_num = 0
            matches, name, match_dist = [], None, None

            t1 = time.time()    
            while True:
                status, frame = cap.read()
                if not status:
                    break
                # If frame comes from webcam, flip it so it looks like a mirror.
                if video_path == 0:
                    frame = cv2.flip(frame, 2)
                
                if frame_num % detection_interval == 0:
                    smaller_frame = convert_to_rgb(
                                        cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
                    
                    matches = self.face_recognizer\
                                    .recognize_faces(
                                        image = smaller_frame, 
                                        threshold=0.6,
                                        bboxes=None)
                    # print(matches)
                    for face_bbox, match, dist in matches:
                        name = match['name'] if match is not None else 'Unknown'
                        match_dist = '{:.2f}'.format(dist) if dist < 1000 else 'INF'
                        name = name + ', Dist: {}'.format(match_dist)
                
                        if verbose:
                            # draw face labels
                            draw_annotation(frame, name, 2 * np.array(face_bbox))
                            print('Match: {}, dist: {}'.format(name, match_dist))
                
                if save_output:
                    video_writer.write(frame)
                
                if preview:
                    cv2.imshow('Preview', cv2.resize(frame, (680, 480)))
                    print('[INFO] Enter q to exit')
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                frame_num += 1
            
            t2 = time.time()

            print('Time:{}'.format((t2 - t1) / 60))
            print('Total frames: {}'.format(frame_num))
            print('Time per frame: {}'.format((t2 - t1) / frame_num))

        except Exception as exc:
            raise exc
        finally:
            cv2.destroyAllWindows()
            cap.release()
            video_writer.release()

    
    def register_face_webcam(self, name=None, detection_interval=5):
        if name is None:
            raise NoNameProvided

        cap = None
        try:
            cap = cv2.VideoCapture(0)
            frame_num = 0

            while True:
                status, frame = cap.read()
                if not status:
                    break
                
                if frame_num % detection_interval == 0:
                    # detect faces
                    bboxes = self.face_detector.detect_faces(image=frame)
                    try:
                        if len(bboxes) == 1: 
                            facial_data = self.face_recognizer \
                                                .register_face(
                                                    image=frame, 
                                                    name=name,
                                                    bbox=bboxes[0])
                            if facial_data:
                                draw_bounding_box(frame, bboxes[0])
                                cv2.imshow('Registered Face', frame)
                                cv2.waitKey(0)
                                print('[INFO]Press any key to continue......')
                                break

                    except Exception as exc:
                        traceback.print_exc(file=sys.stdout)
                frame_num += 1
            
        except Exception as exc:
            raise exc
        finally:
            cv2.destroyAllWindows()
            cap.release()


    def register_face_path(self, img_path, name):
        if not path_exists(img_path):
            raise PathNotFound
        try:
            img = cv2.imread(img_path)
            facial_data = self.face_recognizer \
                                .register_face(
                                    image=convert_to_rgb(img), 
                                    name=name)
            if facial_data:
                print('[INFO] Face regsitered...')
                return True
            return False
        except Exception as exc:
            raise exc


if __name__ == "__main__":         
    import os 

    ob = FaceRecognitionVideo(face_detector='dlib')
    # ob.recognize_face_video(video_path=None, 
    #         detection_interval=1, save_output=True, preview=True)
    # register a face
    # ob.register_face_webcam(name="Susanta")


    #####################################
    # Register faces for videos
    
    ob.register_face_path(img_path='data/sample/sagar2.jpg',name="Sagar")
    ob.register_face_path(img_path='data/sample/suhani.jpg',name="Suhani")
    ob.register_face_path(img_path='data/sample/vidit.jpg',name="Vidit")
    ob.register_face_path(img_path='data/sample/amrutha.jpg',name="Amrutha")
    
    ob.recognize_face_video(video_path='data/test.mkv', 
            detection_interval=1, save_output=True, preview=True)
    
    
    if path_exists('data/facial_data.json'):
        os.remove('data/facial_data.json')
    print('[INFO] Test DB file deleted...')

    ###########################################