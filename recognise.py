from keras.models import load_model
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# models
class_names = ['Circle', 'Fist', 'Nothing', 'Ok', 'Palm',
               'Peace', 'Space', 'Thumb', 'Victory']

model = load_model('handgesture.h5')

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

camera = cv2.VideoCapture(0)
camera.set(10, 200)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.01)

score = 0
top, right, bottom, left = 170, 150, 425, 450

def predict_model(image):
    predict = model.predict(image)
    result = class_names[np.argmax(predict)]
    for i in range(len(predict[0])):
        if predict[0][i] == 1.0:
            print(class_names[i])
    return result

while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    clone = frame.copy()
    (height, width) = frame.shape[:2]
    roi = frame[top:bottom, right:left]

    cv2.imshow("Area Detection", roi)

    img = cv2.resize(roi, (64, 64))
    img = img.reshape(1, 64, 64, 3)
    prediction = predict_model(img)

    k = cv2.waitKey(10)
    if k == ord('q'):
        break

    cv2.putText(clone, prediction, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 3,
                (0, 0, 255), 10, lineType=cv2.LINE_AA)
    cv2.imshow("Hand Gesture Regconition", cv2.resize(clone, dsize=None, fx=0.5, fy=0.5))
    cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)


cv2.destroyAllWindows()
camera.release()
