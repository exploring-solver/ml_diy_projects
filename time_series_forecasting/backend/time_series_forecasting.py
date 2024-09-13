import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load Dataset
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url, index_col='Month', parse_dates=True)

# Train Test Split
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Fit ARIMA Model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Forecasting
forecast = model_fit.forecast(steps=len(test))

# Evaluate the model
error = mean_squared_error(test, forecast)
print(f'Test MSE: {error}')

# Plot forecast vs actual
plt.plot(test, label='Actual')
plt.plot(forecast, label='Forecast')
plt.legend()
plt.show()

# Save model
import pickle
with open('models/time_series_model.pkl', 'wb') as f:
    pickle.dump(model_fit, f)