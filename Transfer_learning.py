import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import PIL.Image as Image
import PIL
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub

"""Load data"""
IMAGE_SHAPE = (224, 224)
classifier = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape=IMAGE_SHAPE+(3,))
])

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

flowers_labels = {
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
        img_resized = cv2.resize(img, (224, 224))
        x.append(img_resized)
        y.append(flowers_labels[flower_name])

"""Convert X and Y into numpy array"""
x = np.array(x)
y = np.array(y)

"""Use sklearn to split the data into x and y train"""
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)

"""scale my x and y train"""
x_train_scale = x_train / 255
x_test_scale = x_test / 255

"""Predict using pretrain classifier"""
pred = classifier.predict(np.array([x[0], x[1], x[3]]))
print(np.argmax(pred, axis=1))

"""retrain the tf model"""
feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False
)

num_of_flowers = 5
model = tf.keras.Sequential([
    pretrained_model_without_top_layer,
    tf.keras.layers.Dense(num_of_flowers)
])

model.summary()

# Compile Model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
model.fit(x_train_scale, y_train, epochs=5)
print()
model.evaluate(x_test_scale, y_test)