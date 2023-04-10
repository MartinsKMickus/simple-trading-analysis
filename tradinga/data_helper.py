import datetime
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import tradinga.constants as constants
from tradinga.api_helper import alpha_vantage_intraday_extended

DATA_DIR = constants.DATA_DIR
STOCK_DIR = constants.STOCK_DIR


# Create directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
if not os.path.exists(STOCK_DIR):
    os.makedirs(STOCK_DIR)


# Saves symbol data to a file.
# If file exists, then it will be updated with newest data.
# symbol = None saves symbol list file
def save_data_to_csv(symbol: str, data: pd.DataFrame, interval: str):
    # Remove empty values
    data.dropna(how='all', inplace=True)

    if symbol == None:
        file_path = os.path.join(DATA_DIR, 'All_symbols.csv')
        data.to_csv(file_path, index=False)
        return

    file_path = os.path.join(STOCK_DIR, f'{symbol}_{interval}.csv')
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
def load_existing_data(symbol: str, interval: str):
    data = None
    if symbol == None:
        file_path = os.path.join(DATA_DIR, 'All_symbols.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            # Remove empty values
            data.dropna(how='all', inplace=True)
    else:
        file_path = os.path.join(STOCK_DIR, f'{symbol}_{interval}.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            # Remove empty values
            data.dropna(how='all', inplace=True)
    return data


# Download newest data for symbol
def download_newest_data(symbol: str, interval: str):
    # Get local data and check what has to be downloaded
    existing_data = load_existing_data(symbol, interval)
    if existing_data is not None and not existing_data.empty:
        last_date = pd.to_datetime(existing_data['time'].max())
    else:
        last_date = None

    # Download and save
    data = alpha_vantage_intraday_extended(symbol, interval, last_date)
    if isinstance(data, pd.DataFrame):
        save_data_to_csv(symbol, data, interval)


def get_data_interval(data: pd.DataFrame, date_from: datetime.datetime = None, date_to: datetime.datetime = None) -> pd.DataFrame:
    """
    Takes data and returns new data object at timeframe

    Args:
        data (pandas.DataFrame)
        date_from (datetime.datetime)
        date_to (datetime.datetime)

    Returns:
        Filtered data
    """
    # Convert possible string values to datetime
    data['time'] = pd.to_datetime(data['time'])
    filtered_data = data.copy()
    if date_from is not None:
        filtered_data = filtered_data[(filtered_data['time'] > date_from)]
    if date_to is not None:
        filtered_data = filtered_data[(filtered_data['time'] < date_to)]
    
    return filtered_data


def scale_and_sort(data: pd.DataFrame):
    """
    Takes data and returns new scaled and sorted data object

    Args:
        data (pandas.DataFrame)

    Returns:
        Scaled and filtered data
    """
    data.sort_values('time', inplace=True)
    data['time'] = pd.to_datetime(data['time'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(data['close'].values.reshape(-1, 1))