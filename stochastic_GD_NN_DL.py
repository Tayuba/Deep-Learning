import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random

# Load Data sets in dataframe
df = pd.read_csv("home_price.csv")
print(df)

# Scale min max
scale_x = preprocessing.MinMaxScaler()
scale_y = preprocessing.MinMaxScaler()
x_scale = scale_x.fit_transform(df.drop("price", axis=1))
print(x_scale)
y_scale = scale_y.fit_transform(df.price.values.reshape(df.shape[0], 1))
print(y_scale)

# Implement stochastic gradient descent
def stochastic_gradient_descent(x, y_true, epochs, learning_rate= 0.01):
    number_of_features = x.shape[1]
    w = np.ones(shape=(number_of_features))
    b = 0
    total_sample = x.shape[0]

    cost_list = []
    epochs_list = []

    for i in range(epochs):
        random_index = random.randint(0, total_sample - 1)
        x_rand = x[random_index]
        y_rand = y_true[random_index]
        y_predicted = np.dot(w, x_rand.T) + b

        w_grad = -(2 / total_sample) * (x_rand.T.dot(y_rand - y_predicted))
        b_grad = -(2 / total_sample) * (y_rand - y_predicted)

        w = w - learning_rate * w_grad
        b = b - learning_rate * b_grad

        cost = np.mean(np.square(y_rand - y_predicted))

        if i % 10 == 0:
            cost_list.append(cost)
            epochs_list.append(i)

    return w, b, cost, cost_list, epochs_list

w, b, cost, cost_list, epoch_list = stochastic_gradient_descent(x_scale, y_scale.reshape(y_scale.shape[0],), 10000)
print(w, b, cost)

# Plot graph to see the reductions of cost
plt.xlabel("epoch")
plt.ylabel("cost")
plt.plot(epoch_list, cost_list)
plt.show()

# Create prediction function
def predict(area, bedrooms, w, b):
    x_scale = scale_x.transform([[area, bedrooms]])[0]

    scaled_price = w[0] * x_scale + w[1] * x_scale[1] + b
    actual_price = scale_y.inverse_transform([scaled_price])[0][0]
    print(actual_price)
    return actual_price
#
predict(2600, 4, w, b)