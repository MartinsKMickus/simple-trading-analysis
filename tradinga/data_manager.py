

import os
import pandas as pd

from tradinga.settings import DATA_DIR, STOCK_DIR


class DataManager():
    def __init__(self) -> None:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        if not os.path.exists(STOCK_DIR):
            os.makedirs(STOCK_DIR)

    def preprocess_data(data, date_from = None, date_to = None):
        data.dropna(how='all', inplace=True)
        data.columns = map(str.lower, data.columns)
        data['time'] = pd.to_datetime(data['time'])
        # filtered_data = data.copy()
        # if date_from is not None:
        #     filtered_data = filtered_data[(filtered_data['time'] > date_from)]
        # if date_to is not None:
        #     filtered_data = filtered_data[(filtered_data['time'] < date_to)]
        # Removing timezone info
        idx = data.index
        idx = idx.tz_localize(None)
        data.index = idx
        data.reset_index(inplace=True)
        data = data.sort_values(by='Datetime', ascending=False)