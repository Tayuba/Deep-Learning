import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow_addons import losses

""" Data exploration"""
df = pd.read_csv("Churn.csv")
print(df)

# Drop customer ID
df.drop("customerID", axis=1, inplace=True)

# Convert my total charges to float
pd.to_numeric(df.TotalCharges, errors="coerce").isnull()

# Checking nan in TotalChargees
print(df[pd.to_numeric(df.TotalCharges, errors="coerce").isnull()])

# Dropping nan in TotalCharges
df = df[df.TotalCharges!=" "]
print(df.shape)

# Convert my total charges to float
df.TotalCharges = pd.to_numeric(df.TotalCharges)
print(df.TotalCharges.values)

# Check stats
print(df)

# View data
df.hist(bins=50, figsize=(15, 10))
# plt.show()

# Check Correlation of features
print(df.corr(), "\n")
plt.figure(figsize=(10, 5))
sn.heatmap(df.corr(), annot=True)
# plt.show()

# Checking customers who are leaving and those who are not leaving
tunure_chorn_no = df[df.Churn=="No"].tenure
tunure_chorn_yes = df[df.Churn=="Yes"].tenure

# Plot leaving and not leaving customers
plt.hist([tunure_chorn_no, tunure_chorn_yes], color=["green", "red"], label=["Churn No", "Churn Yes"])
plt.xlabel("Tenure")
plt.ylabel("Number of Customers")
plt.title("Customer Churn Prediction Visualization")
plt.legend()
# plt.show()

mc_chorn_no = df[df.Churn=="No"].MonthlyCharges
mc_chorn_yes = df[df.Churn=="Yes"].MonthlyCharges

# Plot leaving and not leaving customers
plt.hist([mc_chorn_no, mc_chorn_yes], color=["green", "red"], label=["Churn No", "Churn Yes"])
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")
plt.title("Customer Churn Prediction Visualization")
plt.legend()
# plt.show()

# Checking unique object to in the dataframe
def unique_values(dataframe):
    for column in dataframe:
        if dataframe[column].dtypes =="object":
            print(f"{column}: {dataframe[column].unique()}")

unique_values(df)

'''From the unique values, i observe "No internet service" and "No phone service" is the same as "No",
I will therefore replace it with No'''
df.replace("No internet service", "No", inplace=True)
df.replace("No phone service", "No", inplace=True)


"""I will also replace all 'Yes' and 'No' values with '1' and '0'"""
def zero_one(dataframe):
    for col in dataframe:
        if dataframe[col].any() == "Yes" or dataframe[col].any() == "No":
            df.replace({"No" : 0, "Yes":1},inplace=True)

zero_one(df)

# Lastly replace male and female with 1 and O
df.replace({"Male" : 0, "Female":1},inplace=True)

for column in df:
    print(f"{column}: {df[column].unique()}")

"""Finally since there are more than two categories in internetservice, contract, and paymentMethod, I will perform One
Hot Encoding on them using pandas get_dummies"""
df = pd.get_dummies(data=df, columns=["InternetService", "Contract", "PaymentMethod"])
des = df.columns
print(des, "\n")
print(df.dtypes)

"""tenure, monthlycharges and totalcharges need to be scaled"""
cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']
scale = MinMaxScaler()
df[cols_to_scale] = scale.fit_transform(df[cols_to_scale])

# Create x and y
x = df.drop("Churn", axis="columns")
y = testLabels = df.Churn.astype(np.float32)

# split the data sets into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15, stratify=y)
# print(df)
print(x_train.shape)


"""Creating Artificial Neural Networks Model"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(26, input_dim=(26), activation="relu"),
    tf.keras.layers.Dense(20, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(5, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

# Compile Model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train my model
model.fit(x_train, y_train, epochs=100)

# Evaluate the model
print(model.evaluate(x_test, y_test))

# Model predict
y_pred = model.predict(x_test)



# function of y predicted
predicted = []
def y_predicted(y):
    for yd in y_pred:
        if yd > 0.5:
            predicted.append(1)
        else:
            predicted.append(0)

y_predicted(y_pred)
print(y_test[:10])
print(predicted)

# Classification report
print(classification_report(y_test, predicted))