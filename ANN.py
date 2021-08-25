import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load data sets
(x_train, y_train), (x_test,y_test) = tf.keras.datasets.cifar10.load_data()

# Image Class
im_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Image show function
def image_show(img):
    plt.figure(figsize= (5, 2))
    plt.imshow(x_train[img])
    # plt.show()
image_show(6)

# Scale the image
x_train_scale = x_train / 255
x_test_scale = x_test / 255

# One Hot Encoding
def One_Hot_Encoder(data):
    cat = tf.keras.utils.to_categorical(
        data, num_classes=10, dtype="float32"
    )
    return cat


y_train_cat = One_Hot_Encoder(y_train)
y_test_cat = One_Hot_Encoder(y_test)
print(y_train_cat, "\n")
print(y_test_cat)


# Train Model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    tf.keras.layers.Dense(3072, activation="relu"),
    tf.keras.layers.Dense(2000, activation="relu"),
    tf.keras.layers.Dense(1000, activation="relu"),
    tf.keras.layers.Dense(10, activation="sigmoid")
])

# Compile Model
model.compile(
    optimizer="SGD",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train_scale, y_train_cat, epochs=50)

predicted_model = np.argmax(model.predict(x_test_scale[0]))
print(im_classes[(predicted_model)])