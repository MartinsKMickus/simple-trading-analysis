

import json
import os
import random
from matplotlib import pyplot as plt
import tensorflow as tf
from tradinga.ai_manager import AIManager
from tradinga.data_manager import DataManager
from tradinga.settings import DATA_DIR, MIN_DATA_CHECKS, STOCK_DIR


class DataAnalyzer:
    data_index = {}

    def __init__(self, analyzer_name:str = 'generic_analyzer', data_dir: str = DATA_DIR, stock_dir: str = STOCK_DIR) -> None:
        self.data_manager = DataManager(data_dir=data_dir, stock_dir=stock_dir)
        self.ai_manager = AIManager(data_dir=data_dir)
        self.data_manager.get_nasdaq_symbols()
        self.analyzer_name = analyzer_name
        self.data_dir = data_dir
        self.interval = '1d'
        self.window = 200
        self.features = 6
        self.min_data_checks = MIN_DATA_CHECKS

    def save_symbol_indices(self):
        """
        Saves symbol indices in class variable for model one-hot encoding

        """
        self.data_index = {symbol: index for index, symbol in enumerate(self.data_manager.symbols)}
        with open(f'{self.data_dir}/{self.analyzer_name}_indeces.json', "w") as file:
            json.dump(self.data_index, file)

    def load_symbol_indices(self):
        """
        Loads existing indices in class variable for model one-hot encoding.
        If not available locally, new indices are created.

        """
        # Load the symbol dictionary from the JSON file
        if not os.path.exists(f'{self.data_dir}/{self.analyzer_name}_indeces.json'):
            self.save_symbol_indices()
        else:
            with open(f'{self.data_dir}/{self.analyzer_name}_indeces.json', "r") as file:
                self.data_index = json.load(file)

    def random_valuation(self, symbol_count = 50):
        if not isinstance(self.data_manager.symbols, list):
            print('No symbols loaded')
            return
        shuffled_list = self.data_manager.symbols.copy()
        random.shuffle(shuffled_list)
        if symbol_count >= len(shuffled_list):
            symbol_count = len(shuffled_list) - 1
        
        metrics = []
        i = 0
        while i < symbol_count:
            loaded_data = self.data_manager.get_symbol_data(symbol=shuffled_list[i], interval=self.interval)
            if len(loaded_data) < self.window + self.min_data_checks:
                print(f'Skipping symbol {shuffled_list[i]}. Not enough data points')
                symbol_count += 1
                i += 1
                continue
            scaled = self.ai_manager.scale_for_ai(data=loaded_data)
            metrics.append([shuffled_list[i], self.ai_manager.get_metrics_on_data(scaled)])
            i += 1
        return metrics

    def random_training(self, symbol_count = 10):
        if not isinstance(self.data_manager.symbols, list):
            print('No symbols loaded')
            return
        if not isinstance(self.ai_manager.model, tf.keras.Model):
            print(f'Creating new model because no model exist at {self.ai_manager.ai_location}')
            self.ai_manager.create_model((self.window, self.features))
            self.ai_manager.save_model()

        shuffled_list = self.data_manager.symbols.copy()
        random.shuffle(shuffled_list)
        if symbol_count >= len(shuffled_list):
            symbol_count = len(shuffled_list) - 1
        
        i = 0
        while i < symbol_count:
            scaled = self.ai_manager.scale_for_ai(data=self.data_manager.get_symbol_data(symbol=shuffled_list[i], interval=self.interval))
            try:
                x_arr, y_arr = self.ai_manager.get_xy_arrays(values=scaled)
            except Exception as e:
                print(f'Skipping symbol {shuffled_list[i]}. Reason:')
                print(e)
                symbol_count += 1
                continue
            self.ai_manager.train_model(x_train=x_arr, y_train=y_arr, log_name=f'Train{shuffled_list[i]}')
            self.ai_manager.save_model()
        