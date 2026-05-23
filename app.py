from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

model = load_model("models/gesture_model.h5")

classes = [
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

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""
    image = ""

    if request.method == "POST":

        file = request.files["image"]

        if file:

            os.makedirs("static", exist_ok=True)

            image = "static/" + file.filename

            file.save(image)

            img = cv2.imread(image)

            img = cv2.resize(img, (64,64))

            img = img / 255.0

            img = np.expand_dims(img, axis=0)

            pred = model.predict(img, verbose=0)

            index = np.argmax(pred)

            prediction = classes[index]

    return render_template(
        "index.html",
        prediction=prediction,
        image=image
    )

if __name__ == "__main__":
    app.run()