import datetime
import os
import unittest
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd

from tradinga.data_manager import DataManager
from tradinga.settings import STOCK_DIR, TESTING_DIR

class DataManagerTests(unittest.TestCase):

    def test_constructor_and_symbol_list(self):
        if os.path.exists(f'{TESTING_DIR}/symbol_test_list.csv'):
            os.remove(f'{TESTING_DIR}/symbol_test_list.csv')
        data_manager = DataManager(data_dir=TESTING_DIR)
        self.assertTrue(os.path.exists(TESTING_DIR))
        data_manager.get_nasdaq_symbols(symbol_file=f'{TESTING_DIR}/symbol_test_list.csv', online=True)
        self.assertTrue(os.path.exists(f'{TESTING_DIR}/symbol_test_list.csv'))
        self.assertIn('AAPL', data_manager.symbols)
        
    def test_data_loading(self):
        if os.path.exists(f'{TESTING_DIR}/{STOCK_DIR}/AAPL_1d.csv'):
            os.remove(f'{TESTING_DIR}/{STOCK_DIR}/AAPL_1d.csv')
        if os.path.exists(f'{TESTING_DIR}/{STOCK_DIR}/AAPL_60m.csv'):
            os.remove(f'{TESTING_DIR}/{STOCK_DIR}/AAPL_60m.csv')
        if os.path.exists(f'{TESTING_DIR}/{STOCK_DIR}/AAPL_90m.csv'):
            os.remove(f'{TESTING_DIR}/{STOCK_DIR}/AAPL_90m.csv')
        data_manager = DataManager(data_dir=TESTING_DIR)
        data = data_manager.get_symbol_data('AAPL', '1d')
        # data_manager.save_data_to_csv(data=data, file_name=f'{TESTING_DIR}/{STOCK_DIR}/AAPL_1d.csv')
        self.assertTrue(os.path.exists(f'{TESTING_DIR}/{STOCK_DIR}/AAPL_1d.csv'))

        data = data_manager.get_symbol_data('AAPL', '60m')
        # data_manager.save_data_to_csv(data=data, file_name=f'{TESTING_DIR}/{STOCK_DIR}/AAPL_60m.csv')
        self.assertTrue(os.path.exists(f'{TESTING_DIR}/{STOCK_DIR}/AAPL_60m.csv'))

        data = data_manager.get_symbol_data('AAPL', '90m')
        self.assertTrue(os.path.exists(f'{TESTING_DIR}/{STOCK_DIR}/AAPL_90m.csv'))
    
    def test_data_update(self):
        if os.path.exists(f'{TESTING_DIR}/{STOCK_DIR}/AAPL_1d.csv'):
            os.remove(f'{TESTING_DIR}/{STOCK_DIR}/AAPL_1d.csv')
        data_manager = DataManager(data_dir=TESTING_DIR)
        date_to = datetime.datetime.now() - datetime.timedelta(days=100)

        data = data_manager.get_symbol_data('AAPL', '1d')
        data = data_manager.filter_data(data=data, date_to=date_to)
        with self.assertRaises(Exception):
            data_manager.save_data_to_csv(data=data, file_name=f'{TESTING_DIR}/{STOCK_DIR}/AAPL_1d.csv')
        saved_data = data_manager.get_symbol_data('AAPL', '1d')
        self.assertTrue(os.path.exists(f'{TESTING_DIR}/{STOCK_DIR}/AAPL_1d.csv'))
        delta = datetime.datetime.now() - pd.to_datetime(saved_data['time'].max())
        # Lets assume that longest market holidays are 5 days
        self.assertTrue(delta < datetime.timedelta(days=5))

        data = data_manager.get_symbol_data('AAPL', '1d', online=True)
        with self.assertRaises(Exception):
            data_manager.save_data_to_csv(data=data, file_name=f'{TESTING_DIR}/{STOCK_DIR}/AAPL_1d.csv')
        saved_data = data_manager.get_symbol_data('AAPL', '1d')
        delta = datetime.datetime.now() - pd.to_datetime(saved_data['time'].max())
        self.assertFalse(saved_data['time'].duplicated().any())

    def test_rsi(self):
        data_manager = DataManager(data_dir=TESTING_DIR)
        data = data_manager.get_symbol_data('AAPL', '1d')
        data_manager.apply_rsi(data=data)
        self.assertIn('rsi', data.columns)

    def test_macd(self):
        data_manager = DataManager(data_dir=TESTING_DIR)
        data = data_manager.get_symbol_data('AAPL', '1d')
        data_manager.apply_macd(data=data)
        self.assertIn('macd', data.columns)
        self.assertIn('macd_histogram', data.columns)
        self.assertIn('macd_signal', data.columns)

    # def test_multiple_interval_data(self): # WRONG IMPLEMENTATION
    #     data_manager = DataManager(data_dir=TESTING_DIR)
    #     data = data_manager.get_symbol_data('AAPL', '60m')
    #     # data_manager.add_additional_interval(data=data, symbol='AAPL', interval='60m')
    #     # for data_col in ('time', 'close', 'open', 'low', 'high', 'volume', 'close_60m', 'open_60m', 'low_60m', 'high_60m', 'volume_60m'):
    #     #     self.assertIn(data_col, data)

    #     data = data.drop('time', axis=1)
    #     # data = data.drop('volume', axis=1)
    #     # data = data.drop('volume_60m', axis=1)
    #     # data = data.drop('volume_30m', axis=1)
    #     plt.plot(data)
    #     plt.show()