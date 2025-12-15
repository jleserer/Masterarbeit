# Window version 10
# python version 3.12.7

#Install the libraries if needed
# !pip install yfinance
# !pip install --upgrade yfinance
# !pip install matplotlib
# !pip install pandas
# !pip install scipy
# !pip install seaborn
# !pip install appdirs
# !pip install lxml>=4.9.1
# !pip install scikit-learn
# !pip install tensorflow

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import yfinance as yfin
from pandas_datareader import data as pdr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Nadam, Adam, SGD, RMSprop
import datetime as dt
from scipy.interpolate import griddata


# Download data for S&P 500
s_and_p_500 = yfin.download("^SPX", start="2021-01-01", end="2024-10-26", interval="1d")

# Check if columns are multi-index and flatten them
if isinstance(s_and_p_500.columns, pd.MultiIndex):
    s_and_p_500.columns = [col[0] for col in s_and_p_500.columns]

# Remove the ticker symbol from the column names
s_and_p_500.columns = [col.replace("^SPX", '') for col in s_and_p_500.columns]

# Convert the index to just the date (remove time and timezone)
s_and_p_500.index = pd.to_datetime(s_and_p_500.index).date

# Display the cleaned DataFrame
print(s_and_p_500)

data = s_and_p_500.copy()

# Define the split dates
# Training set: '2021-01-01' - '2023-06-27'
# Validation set: '2023-06-28' - '2024-01-23'
# Test set: '2024-01-24' - '2024-10-25'

train_size = int(len(data) * 0.65)
validation_size = int(len(data) * 0.15)
test_size = len(data) - train_size - validation_size


# Get the size of different sets
print(train_size)
print(validation_size)
print(test_size)


train_data = data[:train_size]
validation_data = data[train_size:train_size + validation_size]
test_data = data[train_size + validation_size:]



scaler = MinMaxScaler()

# Fit the scaler on training data and transform each dataset separately
train_data_scaled = pd.DataFrame(scaler.fit_transform(train_data), columns=features, index=train_data.index)
validation_data_scaled = pd.DataFrame(scaler.transform(validation_data), columns=features, index=validation_data.index)
test_data_scaled = pd.DataFrame(scaler.transform(test_data), columns=features, index=test_data.index)


# View one dataset
print(train_data_scaled)


# Function to Create Sequences
def create_sequences(data, look_back, target_column=3):  # Target column index for 'Adj Close' is now 3
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i, :])  # Use previous L days' data
        y.append(data[i, target_column])    # Target: Next day's 'Adj Close'
    return np.array(X), np.array(y)



# Test the Create Sequences function with look_back=60
X_test_60,y_test_60=create_sequences(test_data_scaled.values, 60)

#print(X_test_60)
np.info(X_test_60)
#np.info(y_test_60)
#print(X_test_60)


def build_lstm_model(input_shape, 
                     learning_rate=0.001, 
                     optimizer_type='Nadam', 
                     neuron_num_1=256, 
                     neuron_num_2=128, 
                     dense_units=64, 
                     activation='tanh', 
                     dropout_rate=0.2):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(neuron_num_1, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(neuron_num_2, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation=activation))
    model.add(Dense(1))
    
    # Change the optimizer
    if optimizer_type == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)
    elif optimizer_type == 'RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = Nadam(learning_rate=learning_rate)   
        
    model.compile(optimizer=optimizer, loss='mean_squared_error')   
    return model


# Function to tune the LSTM model
def tune_lstm(L_values,
              train_data_scaled,
              validation_data_scaled,
              test_data_scaled,
              learning_rate=0.001,
              optimizer_type='Nadam',
              neuron_num_1=256,
              neuron_num_2=128, 
              dense_units=64, 
              activation='tanh', 
              dropout_rate=0.2,
              batch_size=32):
    
    # Loop through each L value
    for L in L_values:
        # Prepare the data for the current L
        X_train, y_train = create_sequences(train_data_scaled.values, L)
        X_val, y_val = create_sequences(validation_data_scaled.values, L)
        X_test, y_test = create_sequences(test_data_scaled.values, L)

        X_train = tf.convert_to_tensor(X_train)
        y_train = tf.convert_to_tensor(y_train)
        X_val = tf.convert_to_tensor(X_val)
        y_val = tf.convert_to_tensor(y_val)
        X_test = tf.convert_to_tensor(X_test)
        y_test = tf.convert_to_tensor(y_test)

        # Build and train the model
        input_shape = (X_train.shape[1], X_train.shape[2])  # (L, no. of features)
        model = build_lstm_model(input_shape, learning_rate=learning_rate,
                                 optimizer_type=optimizer_type,
                                 neuron_num_1=neuron_num_1,
                                 neuron_num_2=neuron_num_2, 
                                 dense_units=dense_units, 
                                 activation=activation, 
                                 dropout_rate=dropout_rate )

        # Train the model
        model.fit(X_train, y_train, epochs=20, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0)

        # Predict on the test data
        predictions = model.predict(X_test)

        # Inverse transform predictions and true values for 'Adj Close'
        forecast_copy = np.repeat(predictions, 5, axis=-1)
        predictions_rescaled = scaler.inverse_transform(forecast_copy)[:, 0]
        
        y_test_copy = np.repeat(y_test, 5, axis=-1)
        y_test_copy = y_test_copy.reshape(len(y_test), 5)
        y_test_rescaled = scaler.inverse_transform(y_test_copy)[:, 0]

        # Calculate the evaluation metrics

# Tune hyperparameter-learning rate [0.001, 0.0001, 0.00001]
# Test learning rate = 0.001

# Set the seed to ensure reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Initialize lists to store the results
r2_results = []
mse_results = []
mae_results = []

# Tune the lstm model
tune_lstm(L_values, train_data_scaled, validation_data_scaled, test_data_scaled, learning_rate=0.001, optimizer_type='Nadam', neuron_num_1=256,
          neuron_num_2=128, dense_units=64, activation='tanh', dropout_rate=0.2, batch_size=32)

# Clear the previous model and set seeds for reproducibility
def clear_and_reset_tensorflow(seed=42):
    tf.keras.backend.clear_session()
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# Tune hyperparameter-learning rate [0.001, 0.0001, 0.00001]
# Test learning rate = 0.0001


# Clear the previous model and reset seed
clear_and_reset_tensorflow(seed=42)

# Initialize lists to store the results
r2_results = []
mse_results = []
mae_results = []

# Tune the lstm model
tune_lstm(L_values, train_data_scaled, validation_data_scaled, test_data_scaled, learning_rate=0.0001, optimizer_type='Nadam', neuron_num_1=256,
          neuron_num_2=128, dense_units=64, activation='tanh', dropout_rate=0.2, batch_size=32)


# Tune hyperparameter-learning rate [0.001, 0.0001, 0.00001]
# Test learning rate = 0.00001

# Clear the previous model and reset seed
clear_and_reset_tensorflow(seed=42)
