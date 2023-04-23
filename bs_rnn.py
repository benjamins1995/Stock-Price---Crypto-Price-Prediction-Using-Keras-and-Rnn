# -*- coding: utf-8 -*-
"""
 Recurrent Neural Network

 Part 1 - Data Preprocessing

 Importing the libraries
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


""" Importing the training set """

dataset_train = pd.read_csv('stock_data_train.csv')
'''
 THE INDEX: [:, 1:2] use it like for access only
  one col for this case its the open col.'''
training_set = dataset_train.iloc[:, 1:2].values


""" Feature Scaling """
from sklearn.preprocessing import MinMaxScaler
# The Normalisation, recommend when have sigmoid output.
sc = MinMaxScaler(feature_range= (0, 1)) 
training_set_scaled = sc.fit_transform(training_set)

" Creating a data structure with 60 timesteps and 1 output "
X_train = []
y_train = []

''' a -> 60 timesteps give the best result.
    b -> 500 the rows size of the training_set_scaled
    like --> range(a, b). '''

for i in range(60, 500):
    '''
     becuse timesteps is 60, we need to start 
     from 0 this is why we are doing: (i - timesteps) '''
    'Xt'
    X_train.append(training_set_scaled[i - 60: i, 0])
    'Xt+1'
    y_train.append(training_set_scaled[i, 0])
    
    'set the dataFrame to numpy arry'
X_train, y_train = np.array(X_train), np.array(y_train)

""" Reshaping """
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
'must do reshape for 2d -> 3d'

""" Part 2 - Building and Training the RNN

 Importing the Keras libraries and packages
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


""" Initialising the RNN """
regressor = Sequential()


""" Adding the first LSTM layer and some Dropout regularisation """
'''input_shape: You can add dimensions with additional information, for example more columns
    units: how much neurons
    return_sequences when have more then one LSTM you need 
        to return true except the last one.'''
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
'Dropout: must to avoid overfitting 0.2 is good'


""" Adding a second LSTM layer and some Dropout regularisation """
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))


""" Adding a third LSTM layer and some Dropout regularisation """
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))


""" Adding a fourth LSTM layer and some Dropout regularisation  """
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

""" Adding the output layer """
regressor.add(Dense(units = 1))


""" Compiling the RNN """
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


""" Fitting the RNN to the Training set """
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
''' Instead of waiting for the value to go back and forth according 
    to the loss, then we will decide that it will happen every 32 times '''

" Part 3 - Making the predictions and visualising the results "

" Getting the real stock price "
dataset_test = pd.read_csv('stock_data_test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

" Getting the predicted stock price "
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
'''Since we are using jumps (time steps) of 60, this means that the model is trained 
on 60 days in each iteration, so we will set our lower bound to be the first day of 
the test data minus 60, since each training must be 60 days including the first unit.'''
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
'we are working only on input data not on test data, this is why we need to do scaling again'
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
'''now we do the same like above in the x_train
   the range(60, 80) 60 we must its our lower bound & 80 its the size, 
   the test data contine only 20 an addition to 60 its 80'''
for i in range(60, 90):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

""" Getting the predicted stock price """
predicted_stock_price = regressor.predict(X_test)
'to get the data in read mode'
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

"""### Visualising the results"""

plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()




" additional info: "

"""
In our case, Adam is better because RMSE is indeed good for RNN, 
but in our case we are not looking for values close to the real value, 
but prior information, so Adam in this case is better, in case you still 
want to use RMSE, The code is below:
*********************
# import math
# from sklearn.metrics import mean_squared_error
# rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
*********************
Then consider dividing this RMSE by the range of the Google Stock 
Price values of January 2017 (that is around 800) to get a relative error, 
as opposed to an absolute error. It is more relevant since for example 
if you get an RMSE of 50, then this error would be very big if the stock 
price values ranged around 100, but it would be very small if the stock price 
values ranged around 10000.
"""



"""
Improving the RNN
here are different ways to improve the RNN model:
*Getting more training data: we trained our model on the past 5 years 
 of the Google Stock Price but it would be even better to train it on 
 the past 10 years.
*Increasing the number of timesteps: the model remembered the stock 
 prices from the 60 previous financial days to predict the stock price of 
 the next day. That’s because we chose a number of 60 timesteps (3 months). 
*You could try to increase the number of timesteps, by choosing for example 
 120 timesteps (6 months).
*Adding some other indicators: if you have the financial instinct that the stock 
 price of some other companies might be correlated to the one of Google, you 
 could add this other stock price as a new indicator in the training data.
*Adding more LSTM layers: we built a RNN with four LSTM layers but you 
 could try with even more.
*Adding more neurones in the LSTM layers: we highlighted the fact that 
 we needed a high number of neurones in the LSTM layers to respond better 
 to the complexity of the problem and we chose to include 50 neurones in each 
 of our 4 LSTM layers. You could try an architecture with even more neurones 
 in each of the 4 (or more) LSTM layers.
"""
