import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
tf.compat.v1.enable_eager_execution()


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
st.title("📈 Stock Market Predictor with LSTM")
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, RELIANCE.NS):", "AAPL")

if st.button("Predict Stock Price"):
    df = get_stock_data(stock_symbol)
    if df.empty:
        st.error("Invalid Stock Symbol. Please try again.")
    else:
        st.write("### Historical Data:", df.tail())
        
        X, y, scaler = prepare_data(df)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        model = build_lstm_model()
        model.fit(X, y, epochs=5, batch_size=16, verbose=1)
        
        future_inputs = X[-1].reshape((1, 60, 1))
        predicted_prices = []
        for _ in range(7):
            pred_price = model.predict(future_inputs)[0][0]
            predicted_prices.append(pred_price)
            future_inputs = np.append(future_inputs[:, 1:, :], [[[pred_price]]], axis=1)
        
        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Actual Prices'))
        future_dates = pd.date_range(df['Date'].iloc[-1], periods=8, freq='D')[1:]
        fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices.flatten(), mode='lines', name='Predicted Prices'))
        st.plotly_chart(fig)
        
        st.write("### Predicted Prices for Next 7 Days:")
        for i, price in enumerate(predicted_prices.flatten()):
            st.write(f"Day {i+1}: {price:.2f} USD / {price*83:.2f} INR")
