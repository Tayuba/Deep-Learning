import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from sklearn.model_selection import train_test_split

"""Downloading and loading flower data"""
url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file("flower_photos", origin=url, cache_dir=".", untar=True)

"""Convert data to windows path object"""
data_dir = pathlib.Path(data_dir)

"""View list of images"""
# print(list(data_dir.glob("*/*.jpg")))

"""Total number of images"""
image_count = len(list(data_dir.glob("*/*.jpg")))

"""Flower types"""
daisy = list(data_dir.glob("daisy/*"))
dandelion = list(data_dir.glob("dandelion/*"))
roses = list(data_dir.glob("roses/*"))
sunflower = list(data_dir.glob("sunflower/*"))
tulips = list(data_dir.glob("tulips/*"))

"""Show image in PIl"""
img = PIL.Image.open(str(tulips[0]))
plt.imshow(img)
# plt.show()

"""Create a dictionary of all the flower types"""
flowers_images = {
    "daisy ": list(data_dir.glob("daisy/*")),
    "dandelion": list(data_dir.glob("dandelion/*")),
    "roses": list(data_dir.glob("roses/*")),
    "sunflower": list(data_dir.glob("sunflower/*")),
    "tulips": list(data_dir.glob("tulips/*"))
}

flowers_labls = {
    "daisy ": 0,
    "dandelion": 1,
    "roses": 2,
    "sunflower": 3,
    "tulips": 4
}

"""Reading image into open cv and creating X and Y sets"""
x = []
y = []

for flower_name, images in flowers_images.items():
    for image in images:
        img = cv2.imread(str(image))
        img_resized = cv2.resize(img, (180,180))
        x.append(img_resized)
        y.append(flowers_labls[flower_name])

"""Convert X and Y into numpy array"""
x = np.array(x)
y = np.array(y)

"""Use sklearn to split the data into x and y train"""
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

"""scale my x and y train"""
x_train_scale = x_train / 255
x_test_scale = x_test / 255

"""Data Augmentation"""
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.3),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.3),
    tf.keras.layers.experimental.preprocessing.RandomFlip(input_shape=(180, 180, 3))

])

"""Building CNN model"""
number_class = 5
model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding = "same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding = "same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding = "same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(60, activation="relu"),
    tf.keras.layers.Dense(5, activation="softmax")
])

# Compile Model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
model.fit(x_train_scale, y_train, epochs=30)
print()
model.evaluate(x_test_scale, y_test)

