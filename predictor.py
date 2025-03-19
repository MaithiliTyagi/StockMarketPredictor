import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Function to fetch historical stock data
def get_stock_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df.reset_index(inplace=True)
    return df

# Function to prepare data for LSTM
def prepare_data(df, time_steps=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Close']])
    X, y = [], []
    for i in range(time_steps, len(df_scaled)):
        X.append(df_scaled[i-time_steps:i, 0])
        y.append(df_scaled[i, 0])
    return np.array(X), np.array(y), scaler

# Function to build and train LSTM model
def build_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Streamlit UI
st.title("📈 Stock Market Predictor")

# Dropdown to choose between Indian or American Stocks
market_choice = st.selectbox("Select Stock Market Region", ["Indian Stocks", "American Stocks"])

# Define stock symbols for Indian and American stocks
indian_stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS", "LT.NS"]
american_stocks = ["AAPL", "GOOGL", "AMZN", "MSFT", "TSLA", "FB"]

# Dropdown to select a stock symbol based on the selected market
if market_choice == "Indian Stocks":
    stock_symbol = st.selectbox("Select Indian Stock Symbol", indian_stocks)
else:
    stock_symbol = st.selectbox("Select American Stock Symbol", american_stocks)

if st.button("Predict Stock Price"):
    df = get_stock_data(stock_symbol)
    if df.empty:
        st.error("Invalid Stock Symbol. Please try again.")
    else:
        st.write("### Historical Data:", df.tail())
        
        # Prepare the data
        X, y, scaler = prepare_data(df)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build the model
        model = build_lstm_model()
        
        # Train the model
        model.fit(X, y, epochs=5, batch_size=16, verbose=1)
        
        # Predict future prices
        future_inputs = X[-1].reshape((1, 60, 1))
        predicted_prices = []
        for _ in range(7):  # Predict for next 7 days
            pred_price = model.predict(future_inputs)[0][0]
            predicted_prices.append(pred_price)
            future_inputs = np.append(future_inputs[:, 1:, :], [[[pred_price]]], axis=1)
        
        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
        
        # Create plotly chart for actual vs predicted
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Actual Prices'))
        
        # Future dates for prediction
        future_dates = pd.date_range(df['Date'].iloc[-1], periods=8, freq='D')[1:]
        fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices.flatten(), mode='lines', name='Predicted Prices'))
        st.plotly_chart(fig)
        
        # Display predicted prices
        st.write("### Predicted Prices for Next 7 Days:")
        for i, price in enumerate(predicted_prices.flatten()):
            st.write(f"Day {i+1}: {price:.2f} USD / {price*83:.2f} INR")  # Assuming INR conversion rate
