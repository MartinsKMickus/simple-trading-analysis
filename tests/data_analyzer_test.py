import datetime
import os
import shutil
import unittest

import pandas as pd
from tradinga.data_analyzer import DataAnalyzer

from tradinga.data_manager import DataManager
from tradinga.settings import STOCK_DIR, TESTING_DIR

class DataTests(unittest.TestCase):

    def test_constructor(self):
        if os.path.exists(f'{TESTING_DIR}'):
            shutil.rmtree(f'{TESTING_DIR}', ignore_errors=True)
            # os.remove(f'{TESTING_DIR}')
        DataAnalyzer(data_dir=TESTING_DIR)
        self.assertTrue(os.path.exists(f'{TESTING_DIR}/{STOCK_DIR}'))

    def test_data_indices(self):
        data_analyzer = DataAnalyzer(data_dir=TESTING_DIR)
        data_analyzer.load_symbol_indices()
        data_manager = DataManager(data_dir=TESTING_DIR)
        data_manager.get_nasdaq_symbols(symbol_file=f'{TESTING_DIR}/symbol_test_list.csv')
        for symbol in range(len(data_manager.symbols)):
            self.assertEqual(symbol, data_analyzer.data_index[data_manager.symbols[symbol]])