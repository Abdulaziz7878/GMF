# Time Series Analysis and Forecasting of TSLA, BND, and SPY

This repository contains a comprehensive analysis and forecasting of stock prices for Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and SPDR S&P 500 ETF Trust (SPY). The analysis employs various methodologies including data acquisition, cleaning, exploratory data analysis (EDA), statistical testing for stationarity, and forecasting using ARIMA, SARIMA, and LSTM models.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Acquisition](#data-acquisition)
- [Data Cleaning](#data-cleaning)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Stationarity Testing](#stationarity-testing)
- [Time Series Decomposition](#time-series-decomposition)
- [Risk Metrics Calculation](#risk-metrics-calculation)
- [Forecasting Models](#forecasting-models)
- [Results Interpretation](#results-interpretation)
- [Conclusion](#conclusion)

### Installation

To run the code in this repository, ensure you have the following libraries installed:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn tensorflow yfinance
```

### Usage

Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

Prepare your environment by installing the required libraries.
Run the analysis script to perform time series forecasting:

```bash
python analysis_and_modeling.py
```

### Data Acquisition

The analysis begins with acquiring historical stock price data using the yfinance library. This library allows users to download market data directly from Yahoo Finance.
Tickers: The selected assets for analysis are Tesla (TSLA), BND (Vanguard Total Bond Market ETF), and SPY (S&P 500 ETF).
Date Range: The data spans from January 1, 2015, to October 31, 2024.

### Data Cleaning

After acquiring the data, it is essential to clean it to handle any missing values and ensure consistency. The cleaning process involves:
Interpolation: Filling missing values using linear interpolation.
Backward Fill: Any remaining missing values are filled using backward fill methods.

### Exploratory Data Analysis (EDA)

EDA helps in understanding the dataset's characteristics through summary statistics and visualizations. Key components include:
Summary Statistics: Count, mean, standard deviation, minimum, maximum, and quartiles for each ticker.

| Ticker | Count | Mean   | Std Dev | Min    | 25%    | Median | 75%    | Max    |
| ------ | ----- | ------ | ------- | ------ | ------ | ------ | ------ | ------ |
| BND    | 2474  | 70.09  | 4.89    | 62.64  | 66.31  | 68.89  | 73.81  | 79.81  |
| SPY    | 2474  | 310.25 | 111.25  | 157.33 | 214.82 | 275.81 | 402.30 | 584.59 |
| TSLA   | 2474  | 111.44 | 110.12  | 9.58   | 17.07  | 25.04  | 216.87 | 409.97 |

### Stationarity Testing

To ensure that the time series data is suitable for forecasting models like ARIMA and SARIMA, stationarity tests were conducted using the Augmented Dickey-Fuller test.
Results Summary:
TSLA: ADF Statistic: -10.09, p-value: (1.14 \times 10^{-17}) (Stationary)
BND: ADF Statistic: -9.74, p-value: (8.42 \times 10^{-17}) (Stationary)
SPY: ADF Statistic: -10.17, p-value: (7.13 \times 10^{-18}) (Stationary)

### Time Series Decomposition

Time series decomposition is performed on TSLA's adjusted closing prices to analyze its trend, seasonality, and residuals using the seasonal decomposition method.

### Risk Metrics Calculation

Value at Risk (VaR) and Sharpe Ratio are computed to evaluate risk associated with each asset.
Results Summary:
VaR indicates potential losses under normal market conditions.

### Forecasting Models

Forecasting was conducted using three different models—ARIMA, SARIMA, and LSTM—each chosen for their strengths in handling time series data.
ARIMA Model: Used for forecasting based on historical data.
SARIMA Model: Extends ARIMA by incorporating seasonality.
LSTM Model: Captures long-term dependencies in sequential data.

### Results Interpretation

The results from each forecasting model were analyzed to determine expected trends over the forecast duration:
Summary of Findings:
Expected Trend: For TSLA, both ARIMA and SARIMA predicted an upward trend.
Confidence Intervals: Indicated ranges of possible price fluctuations.

### Conclusion

This comprehensive analysis encapsulates a robust approach to time series forecasting using Python's powerful libraries such as pandas, statsmodels, and tensorflow. By employing various models—ARIMA, SARIMA, and LSTM—investors can gain insights into potential future price movements of assets like TSLA, BND, and SPY.
The methodologies outlined provide a solid foundation for financial analysis and decision-making in volatile markets while emphasizing the importance of thorough data preparation and exploratory analysis prior to modeling efforts. For further questions or contributions to this project, please contact the repository maintainer or open an issue on GitHub!
