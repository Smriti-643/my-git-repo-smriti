import cv2
from deepface import DeepFace

# Load the face cascade classifier
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open a connection to the webcam
cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze the frame for emotions
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

        if faces is not None and len(faces) > 0:
            # If faces are detected, draw rectangles around them and put text for emotion
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract the dictionary from the list
            if result:
                prediction_dict = result[0]

                # Access the "dominant_emotion"
                dominant_emotion = prediction_dict["dominant_emotion"]

                # Print the result
                print([dominant_emotion])

                # Prepare the text for cv2.putText
                text = f"{dominant_emotion}"

                # Define font
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Put the text on the image (positioned relative to the first detected face)
                cv2.putText(frame, text, (faces[0][0], faces[0][1] - 10), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

    except ValueError as e:
        print(e)

    # Display the frame
    cv2.imshow('Original Video', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
