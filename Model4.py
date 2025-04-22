import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('JPStockPredict.csv')

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Sort by date
df = df.sort_values(by='date')

# Drop rows with missing values
df = df.dropna()

# Use numerical features to predict 'Close'
features = ['high', 'low', 'volume']  # drop 'open' since it's all zero
target = 'close'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.5f}")
print(f"Model Accuracy: {r2 * 100:.2f}%")

# Offset predicted line for visibility
offset = 0.5  # small vertical shift
y_pred_offset = y_pred + offset

# Plot actual vs predicted (with offset)
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Close', alpha=0.8)
plt.plot(y_pred_offset, label='Predicted Close (offset)', alpha=0.8)
plt.title('Actual vs Predicted Closing Prices (Offset for Visibility)')
plt.xlabel('Sample Index')
plt.ylabel('Closing Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Residuals plot
residuals = y_test.values - y_pred
plt.figure(figsize=(12, 5))
plt.plot(residuals, marker='o', linestyle='', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals (Actual - Predicted Closing Prices)')
plt.xlabel('Sample Index')
plt.ylabel('Residual')
plt.grid(True)
plt.tight_layout()
plt.show()
