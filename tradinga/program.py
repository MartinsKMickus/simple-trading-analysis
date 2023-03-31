import os
import io
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse

import constants

DATA_DIR = constants.DATA_DIR
INTERVAL = constants.INTERVAL
SYMBOLS = constants.SYMBOLS

# Create directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# Saves symbol data to a file.
# If file exists, then it will be updated with newest data.
def save_data_to_csv(symbol, data):
    file_path = os.path.join(DATA_DIR, f'{symbol}_{INTERVAL}.csv')
    try:
        # Read in the existing data (if any) from the file
        existing_data = pd.read_csv(file_path)

        # Concatenate the new data with the existing data
        all_data = pd.concat([data, existing_data], ignore_index=True)

        # Write all the data back to the file in append mode
        all_data.to_csv(file_path, index=False)
    except:
        data.to_csv(file_path, index=False)


# Loads local data of a symbol and returns DataFrame.
# If there is no data then None is returned.
def load_existing_data(symbol):
    file_path = os.path.join(DATA_DIR, f'{symbol}_{INTERVAL}.csv')
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None


# Alpha Vantage API call for intraday extended function.
# Returns DataFrame of symbol with interval 1, 5, 15, 30 or 60 minutes from given date.
# Doesn't look past 2 years from now
# wait_between_calls waits before doing next API call if there is limitation
def alpha_vantage_intraday_extended(symbol, interval, from_date, wait_between_calls = 12):
    if interval not in ("1min", "5min", "15min", "30min", "60min"):
        raise Exception(
            "Wrong function used, supported only 1min, 5min, 15min, 30min, 60min")

    years = 2
    months = 12

    # Get difference from two dates
    if from_date:
        delta = relativedelta(datetime.now(), from_date)
        years = delta.years
        months = delta.months
        if years == 0 and months == 0:
            months = 1
            years = 1

    data = None

    # Get data from oldest
    for year in range(years, 0, -1):
        for month in range(months, 0, -1):
            string = f"year{year}month{month}"
            print(f'Downloading data for {symbol} at {string}')
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={symbol}&interval={interval}&slice=year{year}month{month}&apikey={constants.ALPHA_VANTAGE_API_KEY}'

            response = requests.get(url)
            response.raise_for_status()

            time.sleep(wait_between_calls)

            if isinstance(data, pd.DataFrame):
                data = pd.concat(
                    [pd.read_csv(io.StringIO(response.text)), data], ignore_index=True)
            else:
                data = pd.read_csv(io.StringIO(response.text))

    data['time'] = pd.to_datetime(data['time'])
    if from_date:
        data = data[(data['time'] > from_date)]

    return data


# Download newest data for symbol
def download_newest_data(symbol):
    # Get local data and check what has to be downloaded
    existing_data = load_existing_data(symbol)
    if existing_data is not None and not existing_data.empty:
        last_date = pd.to_datetime(existing_data['time'].max())
    else:
        last_date = None

    # Download and save
    data = alpha_vantage_intraday_extended(symbol, INTERVAL, last_date)
    if isinstance(data, pd.DataFrame):
        save_data_to_csv(symbol, data)


for symbol in SYMBOLS:
    try:
        download_newest_data(symbol)
    except Exception as e:
        print(f'Error downloading data for {symbol}: {e}')
