import os
import cv2
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

data = []
labels = []

dataset_path = "dataset/LeapGestRecog"

for gesture in os.listdir(dataset_path):

    gesture_path = os.path.join(dataset_path, gesture)

    count = 0

    for img_name in os.listdir(gesture_path):

        # only 100 images per gesture
        if count >= 100:
            break

        img_path = os.path.join(gesture_path, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (64, 64))

        data.append(img)

        label = int(gesture.split('_')[0]) - 1
        labels.append(label)

        count += 1

X = np.array(data) / 255.0
y = to_categorical(labels, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Sequential([

    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation='relu'),

    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=5)

model.save("models/hand_gesture_model.h5")

print("✅ Model Saved Successfully!")