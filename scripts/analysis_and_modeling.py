import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from statsmodels.tsa.stattools import adfuller

# Load historical data for stocks
def fetch_data(stock_symbols, start_date, end_date):
    stock_data = yf.download(stock_symbols, start=start_date, end=end_date)
    return stock_data['Adj Close']

# Basic data cleaning and type checking
def sanitize_data(data_frame):
    missing_entries = data_frame.isnull().sum()
    data_frame = data_frame.interpolate(method='linear').fillna(method='bfill')
    for col in data_frame.columns:
        if not pd.api.types.is_float_dtype(data_frame[col]):
            data_frame[col] = pd.to_numeric(data_frame[col], errors='coerce')
    data_frame = data_frame.fillna(method='bfill').fillna(method='ffill')
    return data_frame, missing_entries

# Generate summary statistics for the dataset
def summarize_data(data_frame):
    return data_frame.describe()

# Normalize the dataset
def standardize_data(data_frame):
    standardized_frame = (data_frame - data_frame.mean()) / data_frame.std()
    return standardized_frame

# Visualize closing prices over time
def visualize_closing_prices(data_frame):
    plt.figure(figsize=(14, 7))
    for col in data_frame.columns:
        plt.plot(data_frame[col], label=col)
    plt.title("Adjusted Closing Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.legend()
    plt.show()

# Calculate daily percentage change and plot volatility
def visualize_daily_percentage_change(data_frame):
    daily_change = data_frame.pct_change().dropna()
    plt.figure(figsize=(14, 7))
    for col in daily_change.columns:
        plt.plot(daily_change[col], label=f'{col} Daily % Change')
    plt.title("Daily Percentage Change")
    plt.xlabel("Date")
    plt.ylabel("Percentage Change")
    plt.legend()
    plt.show()
    return daily_change

# Calculate rolling means and standard deviations for volatility analysis
def visualize_rolling_statistics(data_frame, window_size=20):
    plt.figure(figsize=(14, 7))
    for col in data_frame.columns:
        rolling_mean = data_frame[col].rolling(window=window_size).mean()
        rolling_std = data_frame[col].rolling(window=window_size).std()
        plt.plot(rolling_mean, label=f'{col} {window_size}-Day Rolling Mean')
        plt.plot(rolling_std, linestyle='--', label=f'{col} {window_size}-Day Rolling Std Dev')
    plt.title(f"{window_size}-Day Rolling Mean and Standard Deviation")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

# Outlier detection for extreme returns
def identify_outliers(data_frame, threshold_value=3):
    outlier_data = pd.DataFrame()
    for col in data_frame.columns:
        z_scores = (data_frame[col] - data_frame[col].mean()) / data_frame[col].std()
        outlier_data[col] = np.where(z_scores.abs() > threshold_value, data_frame[col], np.nan)
    return outlier_data

# Decompose the time series to analyze trend, seasonality, and residuals
def decompose_time_series(data_frame, target_column, model_type='multiplicative', freq_period=252):
    decomposition_result = seasonal_decompose(data_frame[target_column], model=model_type, period=freq_period)
    decomposition_result.plot()
    plt.show()
    return decomposition_result

# Calculate Value at Risk (VaR) and Sharpe Ratio
def compute_risk_metrics(data_frame, risk_free_rate=0.02):
    daily_returns = data_frame.pct_change().dropna()
    var_95_value = daily_returns.quantile(0.05)
    sharpe_ratio_value = (daily_returns.mean() - risk_free_rate / 252) / daily_returns.std()
    return var_95_value, sharpe_ratio_value

# Check for stationarity of a time series
def assess_stationarity(data_series):
    test_result = adfuller(data_series)
    print('ADF Statistic:', test_result[0])
    print('p-value:', test_result[1])
    print('Critical Values:')
    for key, value in test_result[4].items():
        print(f' {key}: {value}')
    if test_result[1] <= 0.05:
        print("The series is likely stationary.")
    else:
        print("The series is likely non-stationary.")

# ARIMA Model function
def arima_forecasting(training_data, testing_data):
    try:
        arima_instance = ARIMA(training_data, order=(5, 1, 0))
        arima_fit_instance = arima_instance.fit()
        arima_predictions = arima_fit_instance.forecast(steps=len(testing_data))
        predictions_series = pd.Series(arima_predictions, index=testing_data.index)
        
        mae_arima_value = mean_absolute_error(testing_data, predictions_series)
        rmse_arima_value = np.sqrt(mean_squared_error(testing_data, predictions_series))
        mape_arima_value = np.mean(np.abs((testing_data - predictions_series) / testing_data)) * 100
        
        return mae_arima_value, rmse_arima_value, mape_arima_value
    except ValueError as error:
        print(f"ARIMA model error: {error}")
        return None, None, None

# SARIMA Model function
def sarima_forecasting(training_data, testing_data):
    try:
        training_data = pd.to_numeric(training_data, errors='coerce').dropna()
        testing_data = pd.to_numeric(testing_data, errors='coerce').dropna()
        
        if len(training_data) < 12 or len(testing_data) < 12:
            print("Insufficient data for SARIMA.")
            return None, None, None
        
        sarima_instance = SARIMAX(training_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
        
        sarima_fit_instance = sarima_instance.fit(disp=False)
        sarima_predictions = sarima_fit_instance.forecast(steps=len(testing_data))
        
        mae_sarima_value = mean_absolute_error(testing_data, sarima_predictions)
        rmse_sarima_value = np.sqrt(mean_squared_error(testing_data, sarima_predictions))
        mape_sarima_value = np.mean(np.abs((testing_data - sarima_predictions) / testing_data)) * 100
        
        return mae_sarima_value, rmse_sarima_value, mape_sarima_value
    except Exception as error:
        print(f"SARIMA model error: {error}")
        return None, None, None

# Prepare LSTM Data function
def prepare_lstm_dataset(dataset_values, look_back_period=60):
    scaler_instance = MinMaxScaler(feature_range=(0, 1))
    
    scaled_values = scaler_instance.fit_transform(dataset_values.values.reshape(-1, 1))
    
    X_setups, y_targets = [], []
    
    for i in range(look_back_period, len(scaled_values)):
        X_setups.append(scaled_values[i-look_back_period:i, 0])
        y_targets.append(scaled_values[i, 0])
    
    X_setups_array = np.array(X_setups).reshape((len(X_setups), look_back_period, 1))
    
    y_targets_array = np.array(y_targets)
    
    return X_setups_array, y_targets_array, scaler_instance

# LSTM Model function
def lstm_forecasting(train_setups, test_setups=None,
                    look_back_period=60,
                    epochs_count=10,
                    batch_size_count=32):
    
    X_train_setup_array, y_train_target_array, scaler_instance = prepare_lstm_dataset(train_setups,
                                                                                        look_back_period)
    
    lstm_model_instance = Sequential()
    
    lstm_model_instance.add(LSTM(units=50,
                                  return_sequences=True,
                                  input_shape=(X_train_setup_array.shape[1], 1)))
    
    lstm_model_instance.add(LSTM(units=50,
                                  return_sequences=False))
    
    lstm_model_instance.add(Dense(units=1))
    
    lstm_model_instance.compile(optimizer='adam', loss='mean_squared_error')
    
    lstm_model_instance.fit(X_train_setup_array,
                            y_train_target_array,
                            epochs=epochs_count,
                            batch_size=batch_size_count,
                            verbose=1)

    # Return only the model and scaler instance 
    return lstm_model_instance , scaler_instance 

# Forecast using ARIMA model 
def forecast_using_arima(train_setups,
                          forecast_duration=180,
                          order_settings=(5 , 1 , 0)):
    
    arima_model_instance = ARIMA(train_setups , order=order_settings)
    fitted_arima_model_instance = arima_model_instance.fit()
  
    forecasted_values_arima = fitted_arima_model_instance.get_forecast(steps=forecast_duration)
  
    forecasted_mean_values_arima = forecasted_values_arima.predicted_mean 
    confidence_intervals_arima_values = forecasted_values_arima.conf_int() 
  
    return forecasted_mean_values_arima , confidence_intervals_arima_values 

# Forecast using SARIMA model 
def forecast_using_sarima(train_setups,
                            forecast_duration=180,
                            order_settings=(1 , 1 , 1),
                            seasonal_order_settings=(1 , 1 , 1 , 12)):
  
    sarimax_model_instance = SARIMAX(train_setups , 
                                      order=order_settings ,
                                      seasonal_order=seasonal_order_settings)
  
    fitted_sarimax_model_instance = sarimax_model_instance.fit() 

    forecasted_values_sarimax = fitted_sarimax_model_instance.get_forecast(steps=forecast_duration) 

    forecasted_mean_values_sarimax = forecasted_values_sarimax.predicted_mean 
    confidence_intervals_sarimax_values = forecasted_values_sarimax.conf_int() 

    return forecasted_mean_values_sarimax , confidence_intervals_sarimax_values 

# Forecast using LSTM model 
def forecast_using_lstm(lstm_model_instance ,
                            input_dataset ,
                            scaler_used ,
                            look_back_period=60 ,
                            forecast_duration=180 ):

    inputs_for_forecast_generation_initials=input_dataset[-look_back_period:].values.reshape(-1 , 1) 

    inputs_for_forecast_generation_initials=scaler_used.transform(inputs_for_forecast_generation_initials) 

    predicted_forecast_values=[] 

    for _ in range(forecast_duration): 
        X_input_setup=np.array(inputs_for_forecast_generation_initials[-look_back_period:]).reshape(1 , look_back_period , 1) 
        predicted_output=lstm_model_instance.predict(X_input_setup) 
        predicted_forecast_values.append(predicted_output[0 , 0]) 
        inputs_for_forecast_generation_initials=np.append(inputs_for_forecast_generation_initials , predicted_output)[-look_back_period:] 

    # Inverse scale the predicted values 
    predicted_forecast_values=scaler_used.inverse_transform(np.array(predicted_forecast_values).reshape(-1 , 1)).flatten() 

    return predicted_forecast_values 

# Forecast and analyze function 
def forecast_and_evaluate(train_setups ,
                            model_choice="arima" ,
                            forecast_duration=180 ,
                            asset_label="Asset"):

    # Ensure the training dataset index is a DatetimeIndex 
    train_setups.index=pd.to_datetime(train_setups.index) 

    # Generate forecasts with the specified model choice 
    if model_choice.lower()=="arima": 
        predicted_mean_values , confidence_intervals_results=forecast_using_arima(train_setups , forecast_duration) 

    elif model_choice.lower()=="sarima": 
        predicted_mean_values , confidence_intervals_results=forecast_using_sarima(train_setups , forecast_duration) 

    elif model_choice.lower()=="lstm": 
        lstm_model_used , scaler_used=lstm_forecasting(train_setups , train_setups) 
        predicted_mean_values=forecast_using_lstm(lstm_model_used , train_setups , scaler_used , forecast_duration=forecast_duration) 
        confidence_intervals_results=None 

    # Prepare the future index for forecasting 
    future_index_start=train_setups.index[-1] 
    future_index=pd.date_range(start=future_index_start , periods=forecast_duration , freq='D') 

    forecast_series=pd.Series(predicted_mean_values , index=future_index) 

    # Plot historical and future forecasts 
    plt.figure(figsize=(14 , 7)) 
    train_setups.plot(label='Historical Data' , color='blue') 
    forecast_series.plot(label=f'Forecast ({model_choice.upper()})' , color='orange') 

    # Plot confidence intervals if available 
    if confidence_intervals_results is not None: 
        plt.fill_between(future_index ,
                        confidence_intervals_results.iloc[:,0],
                        confidence_intervals_results.iloc[:,1],
                        color='pink' , alpha=0.3) 

    plt.axvline(x=future_index_start,color='gray', linestyle='--', label="Forecast Start") 

    plt.title(f'{model_choice.upper()} Forecast for {asset_label} Stock Prices (Future Prediction)')
    plt.xlabel('Date') 
    plt.ylabel('Price') 
    plt.legend() 
    plt.show() 

    # Print analysis summary  
    print("Forecast Summary:")  

    if(model_choice!='lstm'):  
        trend_direction_analysis="upward" if predicted_mean_values.index[-1]>predicted_mean_values.index[0] else "downward"  

    else:  
        trend_direction_analysis="upward" if predicted_mean_values[-1]>predicted_mean_values[0] else "downward"  

    print(f"Expected trend over the forecast duration: {trend_direction_analysis}")  

    if confidence_intervals_results is not None:  
        print("Confidence intervals show the range of possible price fluctuations.")  
        
        return {"forecast": forecast_series, "confidence_intervals": confidence_intervals}
    print("\nVolatility and Risk Analysis:")  

    if confidence_intervals_results is not None:  
        print("The forecast includes confidence intervals indicating expected price fluctuation ranges.")  

    else:  
        print("Confidence intervals are unavailable for the LSTM model.")  

    print("\nMarket Opportunities and Risks:")  

    if trend_direction_analysis=="upward":  
        print("Potential market opportunity due to an expected price increase.")  

    else:  
        print("Potential market risk due to an expected price decrease.")  

    # Return forecasts for further use  
    return {"forecast":forecast_series ,"confidence_intervals":confidence_intervals_results}


def extract_forecast(forecast_result, model_name, asset_name, train_data, forecast_period=180,):
    if forecast_result is not None:
        print(f"{model_name} Forecast for {asset_name} completed.")
        return forecast_result['forecast']
    else:
        print(f"{model_name} Forecast for {asset_name} could not be generated.")
        return pd.Series([None] * forecast_period, index=pd.date_range(start=train_data.index[-1], periods=forecast_period, freq='D'))

def portfolio_performance(weights, returns, cov_matrix):
    port_return = np.dot(weights, returns)  # Expected portfolio return
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Portfolio risk
    sharpe_ratio = port_return / port_volatility  # Sharpe Ratio
    return port_return, port_volatility, sharpe_ratio

def neg_sharpe_ratio(weights, returns, cov_matrix):
    return -portfolio_performance(weights, returns, cov_matrix)[2]