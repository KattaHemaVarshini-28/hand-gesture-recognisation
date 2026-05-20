import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/hand_gesture_model.h5")

# Gesture labels
gestures = [
    "Palm",
    "L",
    "Fist",
    "Fist Moved",
    "Thumb",
    "Index",
    "OK",
    "Palm Moved",
    "C",
    "Down"
]

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera not opening")
        break

    # Resize image
    img = cv2.resize(frame, (64, 64))

    # Normalize
    img = img / 255.0

    # Expand dimensions
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)

    gesture = gestures[np.argmax(prediction)]

    # Display prediction
    cv2.putText(
        frame,
        "Gesture: " + gesture,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show webcam
    cv2.imshow("Hand Gesture Recognition", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()