from flask import Flask, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

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

@app.route('/')
def home():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        img = cv2.resize(frame, (64,64))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0

        prediction = model.predict(img)
        gesture = gestures[np.argmax(prediction)]

        cv2.putText(
            frame,
            gesture,
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return "Closed"

if __name__ == "__main__":
    app.run(debug=True)