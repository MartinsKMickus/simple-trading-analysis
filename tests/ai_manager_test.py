

import os
import shutil
from time import sleep
import unittest

from matplotlib import pyplot as plt
from tradinga.ai_manager import AIManager
from tradinga.data_manager import DataManager
from tradinga.settings import TESTING_DIR


class DataTests(unittest.TestCase):

    def test_creation_and_scaling(self):
        if os.path.exists(f'{TESTING_DIR}'):
            shutil.rmtree(f'{TESTING_DIR}', ignore_errors=True)
            # os.remove(f'{TESTING_DIR}')
        ai_manager = AIManager(data_dir=TESTING_DIR)
        data_manager = DataManager(data_dir=TESTING_DIR)
        data = data_manager.get_symbol_data('AAPL', '1d')
        print(data)
        data_for_ai = data.drop(labels='time', axis=1)
        data = data.set_index('time')
        scaled = ai_manager.scale_for_ai(data=data_for_ai)
        plt.plot(data)
        plt.show()
        