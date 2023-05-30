

import os
import shutil
from time import sleep
import unittest

from matplotlib import pyplot as plt
from tradinga.ai_manager import AIManager
from tradinga.data_analyzer import DataAnalyzer
from tradinga.data_manager import DataManager
from tradinga.settings import TESTING_DIR


class AIManagerTests(unittest.TestCase):

    def test_creation_and_scaling(self):
        if os.path.exists(f'{TESTING_DIR}'):
            shutil.rmtree(f'{TESTING_DIR}', ignore_errors=True)
            # os.remove(f'{TESTING_DIR}')
        ai_manager = AIManager(data_dir=TESTING_DIR)
        data_manager = DataManager(data_dir=TESTING_DIR)
        data = data_manager.get_symbol_data('AAPL', '1d')
        # data = data.set_index('time')
        data = data.reset_index()
        scaled = ai_manager.scale_for_ai(data=data)
        scaled_back = ai_manager.scale_back_array(values=scaled)
        fig, axs = plt.subplots(2)
        axs[0].plot(data)
        axs[1].plot(scaled_back)
        plt.show(block=False)
        plt.pause(1)
        plt.close(fig)

    def test_model_training(self):
        ai_manager = AIManager(data_dir=TESTING_DIR)
        data_manager = DataManager(data_dir=TESTING_DIR)
        data_analyzer = DataAnalyzer(data_dir=TESTING_DIR)
        data_analyzer.load_symbol_indices()
        data = data_manager.get_symbol_data('AAPL', '1d')
        scaled = ai_manager.scale_for_ai(data=data)
        x_arr, y_arr = ai_manager.get_xy_arrays(scaled)
        ai_manager.epochs = 2
        ai_manager.create_model(shape=x_arr.shape[1:])
        ai_manager.train_model(x_train=x_arr, y_train=y_arr)
        ai_manager.save_model()

    def test_prediction(self):
        ai_manager = AIManager(data_dir=TESTING_DIR)
        data_manager = DataManager(data_dir=TESTING_DIR)
        data_analyzer = DataAnalyzer(data_dir=TESTING_DIR)
        data_analyzer.load_symbol_indices()
        data = data_manager.get_symbol_data('AAPL', '1d')
        scaled = ai_manager.scale_for_ai(data=data)
        # x_arr, _ = ai_manager.get_xy_arrays(scaled)
        ai_manager.load_model()
        value = ai_manager.predict_next_value(scaled)
        self.assertTrue(value > 0 and value < 1)


        