# for taking images from webcam
import cv2

save_loc = r'save_image/1.jpg'
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # WIDTH
cap.set(4, 480)  # HEIGHT

face_cascade = cv2.CascadeClassifier(
    r'haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_eye.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1, 0)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(len(faces))

    # Display the resulting frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x-40, y-70), (x+w+20, y+h+40), (255, 0, 0), 2)
        # save the detected face
        cv2.imwrite(save_loc, roi_color)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
