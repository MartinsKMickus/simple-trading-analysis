import unittest
import os
from tradinga.data_helper import download_newest_data, load_existing_data
from tradinga.ai_manager import analyze_market_stocks, make_model

class AITests(unittest.TestCase):

    def test_model_creation(self):
        download_newest_data('AAPL', '1h')
        data = load_existing_data('AAPL', '1h')
        make_model(data=data, look_back=50, epochs=3, model_path='temp/TEST_MODEL')
        self.assertTrue(os.path.exists('temp/TEST_MODEL'))


    def test_market_analyzer(self):
        download_newest_data('AAPL', '1h')
        data = load_existing_data('AAPL', '1h')
        make_model(data=data, look_back=50, epochs=3, model_path='temp/TEST_MODEL')
        analyze_market_stocks(model_path='temp/TEST_MODEL', input_window=50)