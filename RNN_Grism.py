import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
'''
We want to create numpy array, so we use .iloc[:, 1:2], instead of .iloc[:, 1];
Extra :2 can be used to creates numpy array. .values creates numpy array
'''

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)

x_train, y_train = [], []

for i in range(60, len(training_set_scaled)):
    x_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
    
'''
x_train will be a matrix in the below format
x_train = [ <=============================previous 60
          t60
          t61
          t62
          ..
          ..
          ..
          ..                          ]
'''    

x_train, y_train = np.array(x_train), np.array(y_train)


'''
We are reshaping the and making it 3D array instead of 2D array
Keras input format is (batchsize, timesteps, input_data)
batchsize = x_train.shape[0] = t60, t60, .................
timesteps = x_train.shape[1] = rows (Previous 60)
input_data = can be added if you want to predict stock price.
For ex: Apple <==> Samasung is correlated.
'''
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regressor = Sequential()

'''
LSTM(# of units= # of LSTM cells = they have to be large number (High Dimensionality) to capture trend,
     return_sequences = True if other LSTM is added, input_shape = 2D, because first one will be taken into account(Last 2))
'''

'''Layer1'''
regressor.add(LSTM(units =50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
regressor.add(Dropout(0.2))

'''Layer 2'''
regressor.add(LSTM(units =50, return_sequences = True))
regressor.add(Dropout(0.2))

'''Layer 3'''
regressor.add(LSTM(units =50, return_sequences = True))
regressor.add(Dropout(0.2))

'''Layer 4 Final Layer so no return _sequences; its default value is False'''
regressor.add(LSTM(units =50))
regressor.add(Dropout(0.2))

'''Adding output Layer'''
regressor.add(Dense(units = 1))

''' optimizer can be adam too'''
regressor.compile(optimizer = 'RMSprop', loss='mean_squared_error')

'''
output (y_train)
  ||
  ||
  ||
Hidden Layer
  ||
  ||
  ||
input (x_train)

It compares the value from x_train with y_train (actual value) and sends the error rate.
'''
'''batch_size: batches of stock prices not updating weight every stock price but 32 stocks at a time'''
'''fit(I/p, O/p, epochs, batch_size)'''

regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)

dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# axis = 0 for vertical and axis = 1 for horizontal
inputs = dataset_total[len(dataset_total) - len(dataset_test) -60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
''' We use transform because sc object has already fit on previous data and 
we need to scale on those data only'''

x_test = []
for i in range(60, len(inputs)):
    x_test.append(inputs[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
predicted_stock_price = regressor.predict(x_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Getting back to normal price

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

