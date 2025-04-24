#!/usr/bin/env python
# coding: utf-8

# # LSTM Model

# In[1]:



# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# Load your stock data
df = pd.read_csv("./JPStockPredict.csv")
data = df[['close']].values

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create train-test split
training_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - training_size
train_data = scaled_data[:training_size]
test_data = scaled_data[training_size - 60:]  # Include past 60 points for LSTM window

# Prepare sequences
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i - time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape for LSTM [samples, time steps, features]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Shift train predictions for plotting
train_plot = np.empty_like(scaled_data)
train_plot[:, :] = np.nan
train_plot[time_step:len(train_predict) + time_step, 0] = train_predict[:, 0]

# Shift test predictions for plotting (fixed)
test_plot = np.empty_like(scaled_data)
test_plot[:, :] = np.nan
start_idx = len(scaled_data) - len(test_predict)
test_plot[start_idx:, 0] = test_predict[:, 0]

# Plot actual vs predicted
plt.figure(figsize=(14,6))
plt.title('Stock Price Prediction')
plt.plot(scaler.inverse_transform(scaled_data), label='Actual Price')
plt.plot(train_plot, label='Train Prediction')
plt.plot(test_plot, label='Test Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.close('all')

# In[7]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Accuracy-style score: how many predictions are within a % tolerance
tolerance_percent = 0.05  # 5% tolerance
correct = 0
total = len(y_test_actual)

for i in range(total):
    actual = y_test_actual[i][0]
    predicted = test_predict[i][0]
    tolerance = actual * tolerance_percent
    if abs(actual - predicted) <= tolerance:
        correct += 1

# --- Evaluation Metrics ---
mae = mean_absolute_error(y_test_actual, test_predict)
r2 = r2_score(y_test_actual, test_predict)
mse = mean_squared_error(y_test_actual, test_predict)
rmse = np.sqrt(mse)

print("\nModel Performance on Test Data")
print("===============================")
print(f"Model Accuracy-like Score: {correct} / {total} test predictions within ±{tolerance_percent*100:.0f}% of actual value.")
print(f"R² Score        : {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")



# Add to the end of the script
import pickle
result_data = {
    'y_test': y_test if 'y_test' in locals() else None,
    'y_pred': y_pred if 'y_pred' in locals() else None
}
with open('temp_model_results.pkl', 'wb') as f:
    pickle.dump(result_data, f)
