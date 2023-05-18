import os
import unittest

import pandas as pd
from tradinga.api_helper import get_nasdaq_symbols

from tradinga.data_helper import download_newest_data, load_existing_data, save_data_to_csv
from tradinga.settings import STOCK_DIR

class DataTests(unittest.TestCase):

    def test_download_apple(self):
        if os.path.exists(f'{STOCK_DIR}/AAPL_60m.csv'):
            os.remove(f'{STOCK_DIR}/AAPL_60m.csv')
        if os.path.exists(f'{STOCK_DIR}/AAPL_1d.csv'):
            os.remove(f'{STOCK_DIR}/AAPL_1d.csv')
        download_newest_data('AAPL', '60m')
        download_newest_data('AAPL', '1d')
    
    
    def test_load_apple(self):
        if os.path.exists(f'{STOCK_DIR}/AAPL_60m.csv'):
            os.remove(f'{STOCK_DIR}/AAPL_60m.csv')
        download_newest_data('AAPL', '60m')
        data = load_existing_data('AAPL', '60m')
        self.assertTrue(isinstance(data, pd.DataFrame))


    def test_download_symbols(self):
        symbols = get_nasdaq_symbols()
        # save_data_to_csv(data=symbols)

        
    # def test_upper(self):
    #     self.assertEqual('foo'.upper(), 'FOO')

    # def test_isupper(self):
    #     self.assertTrue('FoO'.isupper())
    #     self.assertFalse('FOo'.isupper())

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

# if __name__ == '__main__':
#     unittest.main()