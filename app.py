import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf  # Importing yfinance
from keras.models import load_model
import streamlit as st
from datetime import datetime

st.title('Stock Price Prediction')

start = datetime(2010,1,1)
end= datetime(2019,12,31)


yf.pdr_override()
user_input = st.text_input('Enter the Stock Ticker', value='AAPL')


# Fetch stock data for AAPL
df = pdr.get_data_yahoo(user_input, start=start, end=end)

#Describing the data

st.subheader('Data from 2010 - 2019')

st.write(df.describe())
st.dataframe(df)

#Visualization 
st.subheader('Closing Price v/s Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price v/s Time Chart with 100 MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price v/s Time Chart with 100 MA & 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


#Split the data into training & Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#Scaling down the data to 0 to 1 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



#Load my model 
model = load_model('keras_model.h5')


#testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

#Scaled down
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test , y_test = np.array(x_test) , np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_


scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Visualization & final Graph

st.subheader('Predictions vs Original')

fig2 = plt.figure(figsize=(12,6)) #figsize is for resolution
plt.plot(y_test,'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)