
import datetime
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tradinga.ai_helper import predict_advanced_next_values, train_advanced_model
from tradinga.data_helper import load_existing_data
from tradinga import constants


def ai_test1(symbol: str = "AAPL", predict_from: datetime = datetime.datetime(2023, 1, 10)):
    data = load_existing_data(symbol, constants.INTERVAL)
    data['time'] = pd.to_datetime(data['time'])
    data.sort_values('time', inplace=True)

    if predict_from is not None:
        analyze_data = data[(data['time'] < predict_from)].copy()
    else:
        analyze_data = data

    predicted = predict_advanced_next_values(analyze_data)

    # last_time = data['time'].iloc[-1] # get the last time in the original data

    predicted = np.squeeze(predicted) # We got rows as data in column not one row
    predicted_df = pd.DataFrame({'time': pd.date_range(start=predict_from, periods=len(predicted)+1, freq='30min')[1:], 'close': predicted})

    # concatenated_data = pd.concat([data, predicted_df], ignore_index=True) # concatenate the two dataframes

    # concatenated_data['time'] = pd.to_datetime(concatenated_data['time'])
    predicted_df['time'] = pd.to_datetime(predicted_df['time']) # Convert time for plt to understand it
    data = data[(data['time'] > predict_from - datetime.timedelta(days=5)) & (data['time'] < predict_from + datetime.timedelta(days=1))]
    # concatenated_data = concatenated_data[(concatenated_data['time'] > datetime.datetime(2023, 3, 29))]
    plt.plot(data['time'], data['close'], color='black', label='Real')
    plt.plot(predicted_df['time'], predicted_df['close'], color='blue')
    plt.show()


ai_test1()