import unittest
import os
import shutil

from tradinga.data_helper import download_newest_data, load_existing_data
from tradinga.ai_manager import analyze_market_stocks, analyze_model_metrics, make_model

class AITests(unittest.TestCase):

    def test_model_creation(self):
        shutil.rmtree('temp/TEST_MODEL', ignore_errors=True)
        download_newest_data('AAPL', '60m')
        data = load_existing_data('AAPL', '60m')
        make_model(data=data, look_back=50, epochs=3, model_path='temp/TEST_MODEL')
        self.assertTrue(os.path.exists('temp/TEST_MODEL'))


    def test_market_analyzer(self):
        shutil.rmtree('temp/ANALYZER_MODEL', ignore_errors=True)
        download_newest_data('AAPL', '60m')
        data = load_existing_data('AAPL', '60m')
        make_model(data=data, look_back=50, epochs=3, model_path='temp/TEST_ANALYZER_MODEL')
        analyze_market_stocks(model_path='temp/TEST_ANALYZER_MODEL', input_window=50)

    
    def test_model_metrics(self):
        shutil.rmtree('temp/TEST_METRICS_MODEL', ignore_errors=True)
        download_newest_data('AAPL', '60m')
        data = load_existing_data('AAPL', '60m')
        make_model(data=data, look_back=50, epochs=3, model_path='temp/TEST_METRICS_MODEL')
        analyze_model_metrics(model_path='temp/TEST_METRICS_MODEL', input_window=50)