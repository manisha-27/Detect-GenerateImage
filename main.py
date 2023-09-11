import cv2

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
image = cv2.imread('abc.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# If a face is detected, convert it to black and white
if len(faces) > 0:
    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]

    # Save the black and white face image
    cv2.imwrite('abc2.jpg', face)
else:
    print("No face detected in the image.")
