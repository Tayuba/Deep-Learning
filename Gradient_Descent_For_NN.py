import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Read CSV File
Insurance_df = pd.read_csv("Insurance.csv")
print(Insurance_df)

# Dividing the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(Insurance_df[["age", "affordibility"]],
                                                    Insurance_df.bought_insurance,test_size=0.2, random_state=42)

# Scale the data
x_train_scale = x_train.copy()
x_test_scale =x_test.copy()
x_train_scale["age"] =x_train_scale["age"] / 100
x_test_scale["age"] = x_test_scale["age"] /100

# Create simple neuron network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(2,), activation="sigmoid", kernel_initializer="one", bias_initializer="zero")
])
# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(x_train_scale, y_train, epochs=5100)

# Evaluate the model
model.evaluate(x_test_scale, y_test)

# Final weights and bias
coef, intercept = model.get_weights()
print(coef, intercept)

"""Now I going to build Neural Network from scratch without using tensorflow"""

# Sigmoid function
def sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))


# Prediction function
def prediction_function(age, affordability):
    weighted_sum = coef[0]*age + coef[1]*affordability + intercept
    return sigmoid(weighted_sum)

# Predicting the model
predicted_model = prediction_function(0.18, 1)
print(predicted_model)


"""Implement gradient descent function from scratch"""

# Log loss function
def log_loss(y_true, y_predicted):

    epsilon = 1 * np.exp(-15)
    y_predicted_new = [max(i, epsilon) for i in y_predicted]
    y_predicted_new = [min(i, 1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new) + (1 - y_true)*np.log(1 - y_predicted_new))

# Sigmoid function in array
def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))

# Gradient descent function
def gradient_descent(age, affordability, y_true, epochs, loss_thresold):
    """This function helps find the weights(w1, w2) and the bias(b)"""
    w1 = w2 = 1
    b = 0
    learning_rate = 0.5
    n = len(age)

    """Number of iterations"""
    for i in range(epochs):
        weighted_sum = w1 * age + w2 * affordability + b
        y_predicted = sigmoid_array(weighted_sum)
        loss = log_loss(y_true, y_predicted)
        w1d = (1/n) * np.dot(np.transpose(age),  (y_predicted - y_true))
        w2d = (1/n) * np.dot(np.transpose(affordability), (y_predicted - y_true))
        bd = np.mean(y_predicted - y_true)

        w1 = w1 - learning_rate * w1d
        w2 = w2 - learning_rate * w2d
        b = b - learning_rate * bd

        print(f"Epochs: {i}, W1: {w1}, W2:{w2}, Bias: {b}, Loss{loss}")

        if loss <= loss_thresold:
            break

    return w1, w2, b

# Calling the function using the data in my csv file
gradient_descent(x_train_scale["age"], x_train_scale["affordibility"], y_train, 100000, 0.4904)
print(coef, intercept)
