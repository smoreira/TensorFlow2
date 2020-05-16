# Developed by Sandro Silva Moreira - moreira.sandro@gmail.com

import os
import numpy as np
import cv2
import random
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# -----------------------------
# opencv initialization

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# -----------------------------
# Neural Model

model = tf.keras.models.load_model('models/medical_masks.h5')

# -----------------------------

emotions = ('Com Mascara', 'Uso Incorreto', 'Sem Mascara')


while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # main window
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # crop detected face
        detected_face = img[int(y):int(y+h), int(x):int(x+w)]
        detected_face = cv2.cvtColor(
            detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
        detected_face = cv2.resize(
            detected_face, (128, 128))  # resize to 128x128

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
        img_pixels /= 255

        stacked_img = np.stack((img_pixels,)*3, axis=-2)

        # store probabilities
        predictions = model.predict(stacked_img)

        # max indexed array 0: Mascara, 1:Mascara Errada, 2:Sem
        max_index = np.argmax(predictions[0])

        emotion = emotions[max_index]

        # write text above rectangle
        cv2.putText(img, emotion, (int(x), int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Detector de Mascara - Covid19', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break

# kill open cv things
cap.release()
cv2.destroyAllWindows()
