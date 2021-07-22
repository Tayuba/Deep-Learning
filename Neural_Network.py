import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Read CSV File
Insurance_df = pd.read_csv("Insurance.csv")
# print(Insurance_df)

# Dividing the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(Insurance_df[["age", "affordibility"]],
                                                    Insurance_df.bought_insurance,test_size=0.2, random_state=42)

# Scale the data
x_train_scale = x_train.copy()
x_test_scale =x_test.copy()
x_train_scale["age"] =x_train_scale["age"] / 100
x_test_scale["age"] = x_test_scale["age"] /100



"""Now I going to build Neural Network from scratch without using tensorflow"""

# Sigmoid function
def sigmoid(x):
    import math
    return 1 / (1 + math.exp(-x))


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

"""I will create neural network from scratch"""
# First create a class of my neural network
class my_Neural_Network:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.b = 0

    # Create a fit method
    def fit(self, x, y, epochs, loss_threshold):
        self.w1, self.w2, self.b = self.gradient_descent(x["age"], x["affordibility"], y, epochs, loss_threshold)


    # Create predict method
    def predict(self, x_test):
        weighted_sum =self.w1 * x_test["age"] + self.w2 * x_test["affordibility"] + self.b
        return sigmoid_array(weighted_sum)

    # Gradient descent method
    def gradient_descent(self, age, affordibility, y_true, epochs, loss_threshold):
        """This function helps find the weights(w1, w2) and the bias(b)"""
        w1 = w2 = 1
        b = 0
        learning_rate = 0.5
        n = len(age)

        """Number of iterations"""
        for i in range(epochs):
            weighted_sum = w1 * age + w2 * affordibility + b
            y_predicted = sigmoid_array(weighted_sum)
            loss = log_loss(y_true, y_predicted)
            w1d = (1/n) * np.dot(np.transpose(age),  (y_predicted - y_true))
            w2d = (1/n) * np.dot(np.transpose(affordibility), (y_predicted - y_true))
            bd = np.mean(y_predicted - y_true)

            w1 = w1 - learning_rate * w1d
            w2 = w2 - learning_rate * w2d
            b = b - learning_rate * bd

            print(f"Epochs: {i}, W1: {w1}, W2:{w2}, Bias: {b}, Loss{loss}")

            if loss <= loss_threshold:
                break

        return w1, w2, b

my_model = my_Neural_Network()
my_model.fit(x_train_scale, y_train, epochs=500, loss_threshold=0.4904)

# predict
print(my_model.predict(x_test_scale))