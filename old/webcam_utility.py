# for taking images from webcam
import cv2
import time
from utility import *
import os.path


def detect_face(database, model):
    save_loc = r'saved_image/1.jpg'
    capture_obj = cv2.VideoCapture(0)
    capture_obj.set(3, 640)  # WIDTH
    capture_obj.set(4, 480)  # HEIGHT

    face_cascade = cv2.CascadeClassifier(
        r'haarcascades/haarcascade_frontalface_default.xml')

    # whether there was any face found or not
    face_found = False

    # run the webcam for given seconds
    req_sec = 3
    loop_start = time.time()
    elapsed = 0

    while(True):
        curr_time = time.time()
        elapsed = curr_time - loop_start
        if elapsed >= req_sec:
            break

        # capture_object frame-by-frame
        ret, frame = capture_obj.read()
        # mirror the frame
        frame = cv2.flip(frame, 1, 0)

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Display the resulting frame
        for (x, y, w, h) in faces:
            # required region for the face
            roi_color = frame[y-90:y+h+70, x-50:x+w+50]
            # save the detected face
            cv2.imwrite(save_loc, roi_color)
            # draw a rectangle bounding the face
            cv2.rectangle(frame, (x-10, y-70),
                          (x+w+20, y+h+40), (15, 175, 61), 4)

        # display the frame with bounding rectangle
        cv2.imshow('frame', frame)

        # close the webcam when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the capture_object
    capture_obj.release()
    cv2.destroyAllWindows()

    img = cv2.imread(save_loc)
    if img is not None:
        face_found = True
    else:
        face_found = False

    return face_found
    

# detects faces in realtime from webcam feed


def detect_face_realtime(database, model, threshold=0.7):
    text = ''
    font = cv2.FONT_HERSHEY_SIMPLEX
    save_loc = r'saved_image/1.jpg'
    capture_obj = cv2.VideoCapture(0)
    capture_obj.set(3, 640)  # WIDTH
    capture_obj.set(4, 480)  # HEIGHT

    face_cascade = cv2.CascadeClassifier(
        r'haarcascades/haarcascade_frontalface_default.xml')
    print('**************** Enter "q" to quit **********************')
    prev_time = time.time()
    while(True):

        # capture_object frame-by-frame
        ret, frame = capture_obj.read()
        # mirror the frame
        frame = cv2.flip(frame, 1, 0)

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Display the resulting frame
        for (x, y, w, h) in faces:
            # required region for the face
            roi_color = frame[y-90:y+h+70, x-50:x+w+50]

            # save the detected face
            cv2.imwrite(save_loc, roi_color)

            # keeps track of waiting time for face recognition
            curr_time = time.time()

            if curr_time - prev_time >= 3:
                img = cv2.imread(save_loc)
                if img is not None:
                    resize_img(image_path=save_loc, save_path=save_loc)

                    min_dist, identity, registered = find_face_realtime(
                        save_loc, database, model, threshold)

                    if min_dist <= threshold and registered:
                        # for putting text overlay on webcam feed
                        text = 'Welcome ' + identity
                        print('Welcome ' + identity + '!')
                    else:
                        text = 'Unknown user'
                        print('Unknown user' + ' detected !')
                    print('distance:' + str(min_dist))
                # save the time when the last face recognition task was done
                prev_time = time.time()

            # draw a rectangle bounding the face
            cv2.rectangle(frame, (x-10, y-70),
                          (x+w+20, y+h+40), (15, 175, 61), 4)
            cv2.putText(frame, text, (50, 50), font, 1.8, (158, 11, 40), 3)

        # display the frame with bounding rectangle
        cv2.imshow('frame', frame)

        # close the webcam when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the capture_object
    capture_obj.release()
    cv2.destroyAllWindows()


# checks whether the input face is a registered user or not
def find_face_realtime(image_path, database, model, threshold):
    # find the face encodings for the input image
    encoding = img_to_encoding(image_path, model)
    registered = False
    min_dist = 99999
    identity = 'Unknown Person'
    # loop over all the recorded encodings in database
    for name in database:
        # find the similarity between the input encodings and claimed person's encodings using L2 norm
        dist = np.linalg.norm(np.subtract(database[name], encoding))
        # check if minimum distance or not
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > threshold:
        registered = False
    else:
        registered = True
    return min_dist, identity, registered
