import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/hand_gesture_model.h5")

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

img = cv2.imread("test.jpg")
img = cv2.resize(img, (64,64))

img = np.expand_dims(img, axis=0)
img = img / 255.0

prediction = model.predict(img)

print("Gesture:", gestures[np.argmax(prediction)])