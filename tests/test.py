import os
import unittest

import pandas as pd
from tradinga.ai_manager import analyze_market_stocks, make_model

from tradinga.data_helper import download_newest_data, load_existing_data

class SimpleTests(unittest.TestCase):

    def test_download_apple(self):
        download_newest_data('AAPL', '1h')
        # download_newest_data('AAPL', '1d')
    
    
    def test_load_apple(self):
        download_newest_data('AAPL', '1h')
        data = load_existing_data('AAPL', '1h')
        self.assertTrue(isinstance(data, pd.DataFrame))


    def test_model(self):
        download_newest_data('AAPL', '1h')
        data = load_existing_data('AAPL', '1h')
        make_model(data=data, look_back=100, epochs=10, model_path='temp/TEST_MODEL')
        self.assertTrue(os.path.exists('temp/TEST_MODEL'))


    def test_market_analyzer(self):
        download_newest_data('AAPL', '1h')
        data = load_existing_data('AAPL', '1h')
        make_model(data=data, look_back=100, epochs=10, model_path='temp/TEST_MODEL')
        analyze_market_stocks(model_path='temp/TEST_MODEL', input_window=100)

        
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