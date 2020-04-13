from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2
import os
import pickle

if not os.path.exists('result'):
    os.mkdir('result')

model = 'pokedex.model'

labelbins = 'lb.pickle'

image_path = 'examples'

print('[INFO] Loading model...')
model = load_model(model)
with open('lb.pickle', 'rb') as f:
    lb = pickle.load(f)

for x, image in enumerate(os.listdir(image_path)):
    img = cv2.imread(f'{image_path}/{image}')
    print(f'[INFO] Loading image: {image}...')
    output = img.copy()

    img = cv2.resize(img, (96, 96))
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    print('[INFO] Classifying image...')
    prob = model.predict(img)[0]
    index = np.argmax(prob)
    label = lb.classes_[index]

    print(f'[INFO] Result: {label}')
    text = f'Label: {label} | Probability: {round(prob[index] * 100, 2)}%'
    output = imutils.resize(output, width=500)
    cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    cv2.imshow("Result", output)
    cv2.imwrite(f'result/output-{x}.png', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv2.destroyAllWindows()
