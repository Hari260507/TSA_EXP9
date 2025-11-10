# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
 

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("/content/Tomato.csv")

# If there's a 'date' column, convert it to datetime — otherwise create a dummy one
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)
else:
    # If no date column exists, just use a numeric index for demonstration
    data.index = pd.date_range(start='2020-01-01', periods=len(data), freq='D')

# Show columns
print("Available columns:", data.columns)

# --- ARIMA Model Function ---
def arima_model(data, target_variable, order):
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]
    
    # Fit ARIMA model
    model = ARIMA(train_data[target_variable], order=order)
    fitted_model = model.fit()
    
    # Forecast
    forecast = fitted_model.forecast(steps=len(test_data))
    
    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data[target_variable], label='Training Data')
    plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
    plt.plot(test_data.index, forecast, label='Forecasted Data', color='orange')
    plt.xlabel('Date')
    plt.ylabel(target_variable)
    plt.title('ARIMA Forecasting for ' + target_variable)
    plt.legend()
    plt.show()
    
    print("Root Mean Squared Error (RMSE):", rmse)

# Use the 'Average' column for forecasting
arima_model(data, target_variable='Average', order=(5,1,0))

```

### OUTPUT:
<img width="1083" height="738" alt="image" src="https://github.com/user-attachments/assets/99c5b7ae-4875-4f18-b6e2-7786a0b7dc4a" />



### RESULT:
Thus the program run successfully based on the ARIMA model using python.
