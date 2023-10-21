import streamlit as st
import pandas as pd
import numpy as np
from prophet.serialize import model_to_json, model_from_json
import plotly.express as px

def load_data():
    df = pd.read_csv('TSLA.csv')
    columns = ['Date', 'Close']
    ndf = pd.DataFrame(df, columns = columns)
    # ndf = ndf.rename(columns = {'Date': 'ds', 'Close': 'y'})
    return ndf

def join_data(df1, df2):
    df1 = df1.loc[df2.shape[0]:, ['ds', 'yhat']]  \
           .rename(columns={'ds': 'Date', 'yhat': 'Close'})
    df = pd.concat([df1, df2], axis=0).reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date'].astype(str), format='ISO8601')
    df = df.sort_values(by='Date', ascending=False).reset_index(drop=True)
    return df


# st.session_state['age'] = 0
with open('serialized_model.json', 'r') as fin:
    m = model_from_json(fin.read())  # Load model

st.title('TSLA Stock Price Predictive Model')

st.slider('Select periods (Day)', min_value=1, max_value=100, value=50, step=1, key='day')
st.write(f"{st.session_state['day']} Days")

future = m.make_future_dataframe(periods=st.session_state['day'], freq='D')
forecast = m.predict(future)

figure = m.plot(forecast, xlabel='ds', ylabel='yhat', include_legend=True)
st.write(figure)

predicted = join_data(forecast, load_data())
st.write(px.line(predicted, x='Date', y='Close'))
st.data_editor(predicted, disabled=True)