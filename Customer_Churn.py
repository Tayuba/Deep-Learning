import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

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
def ANN(x_train, y_train, x_test, y_test, loss, weights, epochs):
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
        loss=loss,
        metrics=["accuracy"]
    )

    # Train my model
    if weights == -1:
        model.fit(x_train, y_train, epochs=epochs)
    else:
        model.fit(x_train, y_train, epochs=epochs, class_weight=weights)

    # Evaluate the model
    print(model.evaluate(x_test, y_test))

    # Model predict
    y_pred = model.predict(x_test)
    y_pred = np.round(y_pred)

    # Classification report
    print("Classification Reports: \n", classification_report(y_test, y_pred))
    return y_pred



ANN(x_train, y_train, x_test, y_test, "binary_crossentropy", -1, 200)


"""Handling Imbalance in datasets, I will use all the various techniques in dealing with imbalance"""

"""First Technique, Under sampling majority class"""
count_class_0, count_class_1 = df.Churn.value_counts()
print(count_class_0) # """checking the number of zeros in Churn"""
print(count_class_1)#  checking the number of ones in Churn

zeros_class = df[df["Churn"] == 0] #Storing all zeros in the dataframe
print(zeros_class)
ones_class = df[df["Churn"] == 1] #Storing all ones in this dataframe"""
print(ones_class)

count_class_0_under = zeros_class.sample(count_class_1) # Under sampling dataframe
new_target_churn = pd.concat([count_class_0_under, ones_class], axis=0) # concatenating zero and one churn by row in the dataframe
print(new_target_churn, "\n")
print(new_target_churn.Churn.value_counts()) # Verifying the size of zero and one

# separating the dataframe into x and y sets
x = new_target_churn.drop("Churn", axis="columns")
y = new_target_churn.Churn

# split the data sets into train and test sets
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.2, random_state=15, stratify=y)
print(y_train1.value_counts())

# Train my model with the new inputs and target values
ANN(x_train1, y_train1, x_test1, y_test1, "binary_crossentropy", -1, 200)


"""Second Technique, Over sampling """
count_class_0, count_class_1 = df.Churn.value_counts()
print(count_class_0) # """checking the number of zeros in Churn"""
print(count_class_1)#  checking the number of ones in Churn

zeros_class = df[df["Churn"] == 0] #Storing all zeros in the dataframe
print(zeros_class)
ones_class = df[df["Churn"] == 1] #Storing all ones in this dataframe"""
print(ones_class)

count_class_1_over = ones_class.sample(count_class_0, replace=True) # Over sampling  the dataframe
print(count_class_1_over.shape)
new_target_churn_over = pd.concat([zeros_class, count_class_1_over], axis=0) # concatenating zero and one churn by row in the dataframe
print(new_target_churn_over.Churn.value_counts()) # Verifying the size of zero and one

# separating the dataframe into x and y sets
x = new_target_churn_over.drop("Churn", axis="columns")
y = new_target_churn_over.Churn

# split the data sets into train and test sets
x_train2, x_test2, y_train2, y_test2 = train_test_split(x, y, test_size=0.2, random_state=15, stratify=y)
print(y_train2.value_counts())

# Train my model with the new inputs and target values
ANN(x_train2, y_train2, x_test2, y_test2, "binary_crossentropy", -1, 200)

"""Third Technique, Over sampling by producing synthetic samples (SMOTE), this method uses KNN algorithms """
# separating the dataframe into x and y sets
x = df.drop("Churn", axis="columns")
y = df.Churn

smote = SMOTE(sampling_strategy="minority")
x_smote, y_smote = smote.fit_resample(x, y) # this resample my dataframe to be balance
print(y_smote.value_counts())

# split the data sets into train and test sets
x_train3, x_test3, y_train3, y_test3 = train_test_split(x_smote, y_smote, test_size=0.2, random_state=15, stratify=y_smote)
print(y_train3.value_counts())

# Train my model with the new inputs and target values
ANN(x_train3, y_train3, x_test3, y_test3, "binary_crossentropy", -1, 200)


"""Fouth Technique, Use of Ensemble with undersampling"""
# separating the dataframe into x and y sets
x = df.drop("Churn", axis="columns")
y = df.Churn

# split the data sets into train and test sets
x_train4, x_test4, y_train4, y_test4 = train_test_split(x, y, test_size=0.2, random_state=15, stratify=y)
print(y_train4.value_counts())

df3 = x_train4.copy()
df3["Churn"] = y_train4

zeros_class = df3[df3["Churn"] == 0] #Storing all zeros in the dataframe
# print(zeros_class)
ones_class = df3[df3["Churn"] == 1] #Storing all ones in this dataframe"""
print(zeros_class.shape,ones_class.shape)

# Function to split majority dataframe into three
def get_ensemble(df_majority, df_minority, start, end):
    df_ensemble = pd.concat([df_majority[start:end], df_minority], axis=0)
    x_train5 = df_ensemble.drop("Churn", axis=1)
    y_train5 = df_ensemble.Churn

    return x_train5, y_train5

x_train5, y_train5 = get_ensemble(zeros_class, ones_class, 0, 1495)
print(x_train5.shape)

# Train my model with the new inputs and target values
x_train5, y_train5 = get_ensemble(zeros_class, ones_class, 0, 1495)
y_pred1 = ANN(x_train5, y_train5, x_test4, y_test4, "binary_crossentropy", -1, 200)

x_train5, y_train5 = get_ensemble(zeros_class, ones_class, 1495, 2990)
y_pred2 = ANN(x_train5, y_train5, x_test4, y_test4, "binary_crossentropy", -1, 200)

x_train5, y_train5 = get_ensemble(zeros_class, ones_class, 2990, 4130)
y_pred3 = ANN(x_train5, y_train5, x_test4, y_test4, "binary_crossentropy", -1, 200)

print(len(y_pred1), len(y_pred2), len(y_pred3))

# Function to get majority vote
def majority_vote(y1, y2, y3):
    f_pred = []
    for i in range(len(y1)):
        vote = y1[i] +y2[i] + y3[i]
        if vote > 1:
            f_pred.append(1)
        else:
            f_pred.append(0)
    print("Classification Reports For Ensemble: \n", classification_report(y_test4, f_pred))
    return f_pred
majority_vote(y_pred1, y_pred2, y_pred3)