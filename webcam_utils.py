# for taking images from webcam
import cv2
import time


def detect_face():
    save_loc = r'saved_image/1.jpg'
    capture_obj = cv2.VideoCapture(0)
    capture_obj.set(3, 640)  # WIDTH
    capture_obj.set(4, 480)  # HEIGHT

    face_cascade = cv2.CascadeClassifier(
        r'haarcascades/haarcascade_frontalface_default.xml')

    # run the webcam for given seconds
    req_sec = 6
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
            cv2.rectangle(frame, (x-10, y-70), (x+w+20, y+h+40), (255, 0, 0), 2)

        # for putting text overlay on webcam feed
        #text = 'Unknown face'
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(frame, text, (50, 50), font, 2, (255, 255, 0), 2)

        # display the frame with bounding rectangle
        cv2.imshow('frame', frame)

        # close the webcam when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the capture_object
    capture_obj.release()
    cv2.destroyAllWindows()


# recognize the user face by checking for it in the database
def recognize_face(image_path, database, model):
    # find the face encodings for the input image
    encoding = img_to_encoding(image_path, model)

    min_dist = 99999
    threshold = 0.6
    # loop over all the recorded encodings in database
    for name in database:
        # find the similarity between the input encodings and claimed person's encodings using L2 norm
        dist = np.linalg.norm(np.subtract(database[name], encoding))
        # check if minimum distance or not
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > threshold:
        print("User not in the database.")
    else:
        print("Hi! " + str(identity) + ", L2 distance: " + str(min_dist))

    return min_dist, identity
