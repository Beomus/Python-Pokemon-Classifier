import matplotlib
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from model.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

matplotlib.use("Agg")

dataset = 'dataset'

EPOCHS = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
IMAGE_SHAPE = (96, 96, 3)

data = []
labels = []

print("[INFO] Loading images...")
img_paths = sorted(list(paths.list_images(dataset)))
random.seed(69)
random.shuffle(img_paths)

# loop over the input images
for img_path in img_paths:
	image = cv2.imread(img_path)
	image = cv2.resize(image, (IMAGE_SHAPE[0], IMAGE_SHAPE[1]))
	image = img_to_array(image)
	data.append(image)

	label = img_path.split(os.path.sep)[-2]
	labels.append(label)

# rescale pixel to [0.0, 1.0]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, random_state=69)

augment = ImageDataGenerator(
		zoom_range=0.2,
		horizontal_flip=True,
		rotation_range=30, 
		width_shift_range=0.1,
		height_shift_range=0.1, 
		shear_range=0.2, 
		fill_mode="nearest")

print("[INFO] Compiling model...")
model = SmallerVGGNet.build(w=IMAGE_SHAPE[1], h=IMAGE_SHAPE[0], d=IMAGE_SHAPE[2], classes=len(lb.classes_))
optimizer = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])


print("[INFO] Training network...")
history = model.fit_generator(
	augment.flow(x_train, y_train, batch_size=BATCH_SIZE),
	validation_data=(x_test, y_test),
	steps_per_epoch=len(x_train) // BATCH_SIZE,
	epochs=EPOCHS, verbose=1)

print("[INFO] Saving model...")
model.save('pokedex.model')
with open('lb.pickle', 'wb') as f:
	f.write(pickle.dumps(lb))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig('plot.png')
