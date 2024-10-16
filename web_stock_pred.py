import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf

st.title("Stock Price Predictor App")

stock = st.text_input("Enter the Stock ID", "GOOG")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-50,end.month,end.day)

google_data = yf.download(stock, start, end)

model = load_model("stock_price_predictor.keras")
st.subheader("Stock Data")
st.write(google_data)

splitting_len = int(len(google_data)*0.7)
x_test = pd.DataFrame(google_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'],google_data,0))

st.subheader('Original Close Price and MA for 200 days')
google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days')
google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'],google_data,1,google_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 } ,
    index = google_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([google_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)
# Future prediction
future_days = st.slider("Select number of days for future prediction:", 1, 60, 30)  # Slider for user to select future days
last_100_days = scaler.transform(google_data[['Close']].tail(100))  # Last 100 days of the dataset for prediction
future_predictions = []

for _ in range(future_days):
    next_pred = model.predict(np.array([last_100_days]))
    future_predictions.append(next_pred)
    last_100_days = np.append(last_100_days[1:], next_pred, axis=0)

# Convert future_predictions to a 2D array before inverse_transform
future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

# Prepare the dataframe for plotting
last_date = google_data.index[-1]
future_dates = pd.date_range(last_date, periods=future_days + 1)[1:]  # Exclude the start date itself
future_plot_data = pd.DataFrame(future_predictions, index=future_dates, columns=['Future Predictions'])

# Plot the future predictions
st.subheader('Future Predictions')
fig = plt.figure(figsize=(15, 6))
plt.plot(google_data['Close'], label='Historical Data')
plt.plot(future_plot_data, label='Future Predictions')
plt.xlabel("Date")
plt.ylabel("Adj Close Price")
plt.title("Google Stock Price Prediction")
plt.legend()
st.pyplot(fig)
