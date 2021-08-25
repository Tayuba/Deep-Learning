import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd

""" Data exploration"""
df = pd.read_csv("Churn.csv")

# Check for empty data
print(df[df.isnull()])

# Drop customer ID
df.drop("customerID", axis=1, inplace=True)
print(df.dtypes)

# Conver my total charges to float
print(pd.to_numeric(df.TotalCharges, errors="coerce"))

# Checking null in Total Charges
null = (pd.to_numeric(df.TotalCharges, errors="coerce")).isnull()
print(df[null])

# Drop all null
df = df[df.TotalCharges !=" "]
print(pd.to_numeric(df.TotalCharges))

# Check stats
print(df.describe())

# View data
df.hist(bins=50, figsize=(15, 10))
plt.show()

# Check Correlation of features
print(df.corr())
plt.figure(figsize=(10, 5))
sn.heatmap(df.corr(), annot=True)
plt.show()