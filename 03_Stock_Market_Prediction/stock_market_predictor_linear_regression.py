# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:22:04 2023

@author: Vishal Mishra
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('TESLA.csv')
X = dataset.iloc[:, 0].values
Y = dataset.iloc[:, 6].values

x = X[::150]
y = Y[::150]

## Plotting the original trend in stock prices vs Time 

fig = plt.figure()
plt.plot(x, y, color = 'blue')
plt.title('Stock Price of Tesla overview over a period of time')
plt.xlabel('Date')
plt.ylabel('Price')
fig.autofmt_xdate()
plt.show()


## Data pre processing stage for the Linear Regression Model


data = pd.read_csv('TESLA.csv')

data['Date'] = pd.to_datetime(data['Date']) 
data['day_of_year'] = data['Date'].dt.day_of_year
X = data[['day_of_year', 'Open', 'High', 'Low', 'Volume']]  
Y = dataset.iloc[:, 6].values

print(X)
print(Y)


## Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)


## Training the Simple Linear Regression model on the Training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)



## Predicting test set results 

y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred),1) ,Y_test.reshape(len(Y_test),1)),1))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(Y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Calculate R-squared (R2) score
r2 = r2_score(Y_test, y_pred)
print(r2)

# Print the coefficients and Y intercept 
print(regressor.coef_)
print(regressor.intercept_)



# Step 6: Make Predictions on New Data
new_data = {'Date': ['10/19/2033'],
            'Open': [144.990005],
            'High': [506.880005],
            'Low': [222.5],
            'Volume': [350000],
            }

new_data['Date'] = pd.to_datetime(new_data['Date'])

new_data['day_of_year'] = new_data['Date'].dayofyear

new_X = pd.DataFrame(new_data, columns=['day_of_year', 'Open', 'High', 'Low', 'Volume'])

predictions = regressor.predict(new_X)
print(predictions)

