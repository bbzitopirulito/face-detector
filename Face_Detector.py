import cv2

# Load pre-trained face data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# capture cam video
# to use a video, set the video file (string)
webcam = cv2.VideoCapture(0)

while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read()

    if frame is not None:
        # Change the image to grayscale
        grayscalled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        face_coordinates = trained_face_data.detectMultiScale(grayscalled_img)

        for x, y, w, h in face_coordinates:
            # Draw rectangles around the faces
            cv2.rectangle(frame, (x, y), (x + w, y+ h),(0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Face Detector', frame)

        # Wait until a key is pressed in order to stop the code and the image keeps being displayed
        # The parameter auto hits a key from the millisec to millisec
        key = cv2.waitKey(1)

        # Stop if Q/q key is pressed
        if key == 81 or key == 113:
            break

    
# Release the video capture obj
webcam.release()