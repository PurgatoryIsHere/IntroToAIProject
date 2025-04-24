#!/usr/bin/env python
# coding: utf-8

# # Stock Price Prediction using GRU
# In this notebook we are demonstrating how to build a GRU model to use to predict daily stock prices using historical stock data of JP Morgan Chase (JPM).

# ## Importing required libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ## Loading and preparing our Dataset

# In[ ]:


# Load and prepare the dataset
df = pd.read_csv('/content/JPStockPredict.csv')

# Convert 'date' column to timezone-aware datetime in UTC
df['date'] = pd.to_datetime(df['date'], utc=True)

# Sort dataset chronologically and reset index
df.sort_values('date', inplace=True)
df.reset_index(drop=True, inplace=True)

# Extract and normalize 'close' prices
close_prices = df['close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(close_prices)

# ## Sequence Prep

# In[ ]:


def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# ## Test-Train Split

# In[ ]:


look_back = 60
X_all, y_all = create_sequences(scaled_close, look_back)

# Find the index where the date is '2020-12-31'
split_date = pd.Timestamp('2020-12-31', tz='UTC')
split_index = df[df['date'] <= split_date].shape[0] - look_back

X_train = X_all[:split_index]
y_train = y_all[:split_index]
X_test = X_all[split_index:]
y_test = y_all[split_index:]

# ## Building and Training the GRU model

# In[ ]:


# Enhanced GRU model with deeper architecture and dropout regularization
gru_model = Sequential()

# First GRU layer with dropout
gru_model.add(GRU(50, return_sequences=True, input_shape=(look_back, 1)))
gru_model.add(Dropout(0.2))

# Second GRU layer with dropout
gru_model.add(GRU(50, return_sequences=True))
gru_model.add(Dropout(0.2))

# Third GRU layer with dropout
gru_model.add(GRU(50, return_sequences=True))
gru_model.add(Dropout(0.2))

# Fourth GRU layer with dropout
gru_model.add(GRU(50))
gru_model.add(Dropout(0.2))

# Output layer
gru_model.add(Dense(1))

# Compile the model
gru_model.compile(optimizer='adam', loss='mean_squared_error')

# Display model summary
gru_model.summary()


# ## Predicting and Evaluating the GRU model

# In[ ]:


gru_model.compile(optimizer='adam', loss='mean_squared_error')
gru_model.fit(X_train, y_train, epochs=10, batch_size=32)

# In[ ]:


predictions = gru_model.predict(X_test)

predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# In[ ]:


r2 = r2_score(actual_prices, predicted_prices)
plt.figure(figsize=(14, 5))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.title(f'Stock Price Prediction using GRU: {r2 * 100:.4f}% Accuracy')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.close('all')


# Add to the end of the script
import pickle
result_data = {
    'y_test': y_test if 'y_test' in locals() else None,
    'y_pred': y_pred if 'y_pred' in locals() else None
}
with open('temp_model_results.pkl', 'wb') as f:
    pickle.dump(result_data, f)
