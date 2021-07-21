import tensorflow as tf
# import Keras
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

# Load the hand write digits into train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(len(x_train), (len(x_test)))

# Check the shape of each individual sample
print(x_train[0].shape)
"""It is show that each individual sample is (28, 28), in terms of numbers is two dimensional array between 0 to 255
pixels values"""
print(x_train[0])
# View the image of the first sample
print(plt.matshow(x_train[0]))
# plt.show()
"""The image shows '5', to compare the x_train image with the y_train, i will print the value of y_train with the same
  index, and it shows '5' too"""
print(y_train[0])

"""I will scale my train and test sets for bette accuracy"""
x_train = x_train / 255
x_test = x_test / 255

# Find the shape of the samples of hand written digits
x_train_shape = x_train.shape
print(x_train_shape)
"""The shape is (60000, 28, 28) the first dimension is 60000, the second and third is image shape there 28*28= 784.
Making the shape as (60000, 784)"""

"""To be able to use the the samples of the hand written digits in neuron network, i have to flatten the matrix using"""
x_train_flat = x_train.reshape(len(x_train), 28*28)
x_test_flat = x_test.reshape(len(x_test), 28*28)
print(x_train_flat.shape)
print(x_test_flat.shape)
"""Now it gives me my flattened shape as (60000, 784) and (10000, 784) for x_train and x_test respectively"""
# Now if i check the first sample, it gives me single array number
print(x_test_flat[0])

"""Now i will create a simple neuron network with two layers, 784 input layers and 10 output layers using keras"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(784,), activation="sigmoid")
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
# Now train the model
model.fit(x_train_flat, y_train, epochs=10)

# Evaluate accuracy on test data set
test_accuracy = model.evaluate(x_test_flat, y_test)
print(test_accuracy)

"""Now i will predict """
# I will view the first element in the digits data sets
plt.matshow(x_test[0])
# plt.show()

# Now predict the first image
predicted_values = model.predict(x_test_flat)
print(predicted_values[0])
# to check the max values from the array in predicted_values
print(np.argmax(predicted_values[0]))

"""To view my model prediction on graph by using confusion matrix to compare predicted and true values"""
predicted_values = [np.argmax(i) for i in predicted_values] # to get the actual values
confusion_m = tf.math.confusion_matrix(labels=y_test, predictions=predicted_values)
print(confusion_m)
# View confusion_m on seaborn
plt.figure(figsize=(10, 10))
sn.heatmap(confusion_m, annot=True)
plt.xlabel("predicted Values")
plt.ylabel("True Values")
plt.show()

"""I will add one more hidden layer to my neural network to improve it performance"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, input_shape=(784,), activation="relu"),
    tf.keras.layers.Dense(10, activation="sigmoid")
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
# Now train the model
model.fit(x_train_flat, y_train, epochs=10)

# Evaluate accuracy on test data set
test_accuracy = model.evaluate(x_test_flat, y_test)
print(test_accuracy)

"""To view my model prediction on graph by using confusion matrix to compare predicted and true values"""
predicted_values = [np.argmax(i) for i in predicted_values] # to get the actual values
confusion_m = tf.math.confusion_matrix(labels=y_test, predictions=predicted_values)
print(confusion_m)
# View confusion_m on seaborn
plt.figure(figsize=(10, 10))
sn.heatmap(confusion_m, annot=True)
plt.xlabel("predicted Values")
plt.ylabel("True Values")
plt.show()