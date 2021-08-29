import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load data sets
(x_train, y_train), (x_test,y_test) = tf.keras.datasets.cifar10.load_data()
print(y_train)
y_train = y_train.reshape(-1,)
# print(y_train[0])

# Image Class
im_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# Image show function
def image_show(x, y, index):
    plt.figure(figsize= (5, 6))
    plt.imshow(x[index])
    plt.xlabel(im_classes[y[index]])
    plt.show()


# image_show(x_train, y_train, 5)

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

# Train Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(3072, activation="relu"),
    tf.keras.layers.Dense(2000, activation="relu"),
    tf.keras.layers.Dense(1000, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile Model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train_scale, y_train_cat, epochs=10)

# predict
y_pred = model.predict(x_test)
y_classes = [np.argmax(element) for element in y_pred]

# evaluate
model.evaluate(x_test)

#classification
print("Classification Reports: \n", classification_report(y_test, y_classes))

y_train = y_train.reshape(-1,)
image_show(x_train, y_train, 1)