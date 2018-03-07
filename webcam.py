# for taking images from webcam
import cv2

save_loc = r'saved_image/1.jpg'
capture_obj = cv2.VideoCapture(0)
capture_obj.set(3, 640)  # WIDTH
capture_obj.set(4, 480)  # HEIGHT

face_cascade = cv2.CascadeClassifier(
    r'haarcascades/haarcascade_frontalface_default.xml')


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
