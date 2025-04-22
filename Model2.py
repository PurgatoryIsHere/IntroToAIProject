import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("JPStockPredict.csv")

# Ensure date column is in datetime format
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df = df.sort_values('date')
    print("Data sorted chronologically by date.")
else:
    print("No date column found. Assuming data is already in chronological order.")
    # Create a date column if it doesn't exist
    df['date'] = pd.date_range(start='2016-01-01', periods=len(df))

# Create time-based features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter

# Create lagged features (previous N days)
for lag in range(1, 6):  # Create 5 days of lag features
    df[f'close_lag_{lag}'] = df['close'].shift(lag)
    df[f'open_lag_{lag}'] = df['open'].shift(lag)
    df[f'high_lag_{lag}'] = df['high'].shift(lag)
    df[f'low_lag_{lag}'] = df['low'].shift(lag)
    df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

# Create rolling window features (moving averages)
df['close_rolling_7'] = df['close'].rolling(window=7).mean()
df['close_rolling_30'] = df['close'].rolling(window=30).mean()
df['volume_rolling_7'] = df['volume'].rolling(window=7).mean()

# Create price momentum features
df['price_momentum_5'] = df['close'].pct_change(periods=5)
df['price_momentum_10'] = df['close'].pct_change(periods=10)

# Drop rows with NaN values (caused by lag/rolling features)
df = df.dropna()
print(f"Data shape after adding features and removing NaNs: {df.shape}")

# Define features and target
# Exclude 'date' from features
features = [col for col in df.columns if col not in ['date', 'close']]
X = df[features]
y = df['close']

# Split data chronologically - first 80% for training, last 20% for testing
split_idx = int(len(df) * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print(f"Training set: {len(X_train)} samples (first 80% chronologically)")
print(f"Testing set: {len(X_test)} samples (last 20% chronologically)")

# Create and train a decision tree regressor
model = DecisionTreeRegressor(max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy (100% - MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
accuracy = 100 - mape

# Print accuracy
print(f"\nOverall Model Accuracy: {accuracy:.2f}%")

# Create a visualization of predictions vs actual values
plt.figure(figsize=(14, 7))

# Plot actual vs predicted
test_dates = df['date'].iloc[split_idx:]
plt.plot(test_dates, y_test.values, 'b-', label='Actual Close Price', linewidth=2)
plt.plot(test_dates, y_pred, 'r--', label='Predicted Close Price', linewidth=1.5)

plt.title(f'Enhanced Stock Price Prediction (Accuracy: {accuracy:.2f}%)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Close Price', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.gcf().autofmt_xdate()  # Rotate date labels
plt.tight_layout()
plt.savefig('enhanced_stock_prediction.png', dpi=300)
plt.show()

# Get feature importance and display top 15
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 15 Feature Importance:")
print(feature_importance.head(15))

# Visualize top 10 features
plt.figure(figsize=(12, 6))
top_features = feature_importance.head(10)
plt.barh(top_features['Feature'], top_features['Importance'])
plt.title('Top 10 Feature Importance', fontsize=14)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()