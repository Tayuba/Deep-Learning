import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report

"""Data exploration"""
df = pd.read_csv("sonar.csv", header=None)
print(df)

# Check shape
print(df.shape)

# Check null
print(df.isna().sum())

# column names
print(df.columns )

"""Divide dataframe into x and y"""
x = df.drop(60, axis=1)
y = df[60]

"""Covert M and R in y to integer. One Hot Encoding with pandas dummy"""
y = pd.get_dummies(y, drop_first=True)
print(y)

"""Split data set into train and test data sets """
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

"""Build Artificial neural network model"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(60, input_dim=60, activation="relu"),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(15, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss= "binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=100, batch_size=8)
print()

"""Evaluate the overfit model"""
model.evaluate(x_test, y_test)

"""Predict model"""
y_pred = model.predict(x_test).reshape(-1)
# print(y_pred)
print()

def round_up(y):
    y_pred = np.round(y)
    print(y_pred)
    return y_pred
# y_pred = round_up(y_pred)
#
# print(y_test[:10])

# classify = classification_report(y_test, y_pred)
# print(classify)

"""Build Artificial neural network model with a droup layer"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(60, input_dim=60, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(15, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss= "binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=100, batch_size=8)
print()

"""Evaluate the overfit model"""
model.evaluate(x_test, y_test)

y_pred = model.predict(x_test).reshape(-1)

def round_up(y):
    y_pred = np.round(y)
    print(y_pred)
    return y_pred

y_pred = round_up(y_pred)

classify = classification_report(y_test, y_pred)
print(classify)