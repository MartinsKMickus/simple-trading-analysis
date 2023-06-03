

import json
import os
import pickle
import random
from matplotlib import pyplot as plt
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tqdm
from tradinga.ai_manager import AIManager
from tradinga.data_manager import DataManager
from tradinga.settings import DATA_DIR, MIN_DATA_CHECKS, STOCK_DIR


class DataAnalyzer:
    data_index = {}

    def __init__(self, analyzer_name:str = 'generic_analyzer', data_dir: str = DATA_DIR, stock_dir: str = STOCK_DIR, window: int = 200) -> None:
        # Settings
        self.analyzer_name = analyzer_name
        self.data_dir = data_dir

        # Additional settings
        self.min_values = [10000, 10000, 10000, 10000, 10000, 10000]
        self.max_values = [0, 0, 0, 0, 0, 0]
        self.except_symbols = []
        settings = self.load_settings()
        self.interval = '1d'
        self.window = window
        self.features = 6
        self.min_data_checks = MIN_DATA_CHECKS
        self.data_manager = DataManager(data_dir=data_dir, stock_dir=stock_dir)
        self.data_manager.get_nasdaq_symbols()
        one_hot = 0 #len(self.data_manager.symbols)

        # AI Manager
        if settings:
            print(f'Settings loaded!\n Applied min: {self.min_values}\n Applied max: {self.max_values}')
            self.ai_manager = AIManager(data_dir=data_dir, model_name=self.analyzer_name, data_min=np.array(self.min_values), data_max=np.array(self.max_values), one_hot_encoding_count=one_hot)
        else:
            self.ai_manager = AIManager(data_dir=data_dir, model_name=self.analyzer_name, one_hot_encoding_count=one_hot)
        self.ai_manager.window = self.window
        self.ai_manager.data_columns = self.features

        # Lats initializations
        self.load_symbol_indices()
        

    def load_settings(self):
        if os.path.exists(f'{self.data_dir}/{self.analyzer_name}_settings.json'):
            with open(f'{self.data_dir}/{self.analyzer_name}_settings.json', "r") as file:
                loaded_data = json.load(file)
                self.min_values, self.max_values, self.except_symbols = loaded_data
            return True
        return False

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

    def get_min_max_values(self, force = False):
        if os.path.exists(f'{self.data_dir}/{self.analyzer_name}_settings.json') and not force:
            print('Settings already stored!!!')
        if not isinstance(self.data_manager.symbols, list):
            print('No symbols loaded')
            return
        
        for symbol in tqdm.tqdm(self.data_manager.symbols):#self.data_manager.symbols:
            loaded_data = self.data_manager.get_symbol_data(symbol=symbol, interval=self.interval)

            value = loaded_data['open'].max()
            if value > self.max_values[0]:
                if value > 1000:
                    print(f'{symbol} price is too high: {value}')
                    self.except_symbols.append(symbol)
                    continue
                self.max_values[0] = loaded_data['open'].max()
            value = loaded_data['high'].max()
            if value > self.max_values[1]:
                self.max_values[1] = loaded_data['high'].max()
            value = loaded_data['low'].max()
            if value > self.max_values[2]:
                self.max_values[2] = loaded_data['low'].max()
            value = loaded_data['close'].max()
            if value > self.max_values[3]:
                self.max_values[3] = loaded_data['close'].max()
            value = loaded_data['adj close'].max()
            if value > self.max_values[4]:
                self.max_values[4] = loaded_data['adj close'].max()
            value = loaded_data['volume'].max()
            if value > self.max_values[5]:
                self.max_values[5] = loaded_data['volume'].max()

            if loaded_data['open'].min() < self.max_values[0]:
                self.min_values[0] = loaded_data['open'].min()
            if loaded_data['high'].min() < self.max_values[1]:
                self.min_values[1] = loaded_data['high'].min()
            if loaded_data['low'].min() < self.max_values[2]:
                self.min_values[2] = loaded_data['low'].min()
            if loaded_data['close'].min() < self.max_values[3]:
                self.min_values[3] = loaded_data['close'].min()
            if loaded_data['adj close'].min() < self.max_values[4]:
                self.min_values[4] = loaded_data['adj close'].min()
            if loaded_data['volume'].min() < self.max_values[5]:
                self.min_values[5] = loaded_data['volume'].min()
        
        # Save variables to a file
        with open(f'{self.data_dir}/{self.analyzer_name}_settings.json', 'w') as file:
            data_min = [float(item) for item in self.min_values]
            data_max = [float(item) for item in self.max_values]
            self.min_values = data_min
            self.max_values = data_max
            json.dump((self.min_values, self.max_values, self.except_symbols), file)

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
        while i < symbol_count and i < len(shuffled_list):
            loaded_data = self.data_manager.get_symbol_data(symbol=shuffled_list[i], interval=self.interval)
            if len(loaded_data) < self.window + self.min_data_checks:
                print(f'Skipping symbol {shuffled_list[i]}. Not enough data points')
                symbol_count += 1
                i += 1
                continue
            scaled = self.ai_manager.scale_for_ai(data=loaded_data)
            metrics.append([shuffled_list[i], self.ai_manager.get_metrics_on_data(scaled, symbol=shuffled_list[i])])
            i += 1
        return metrics

    def random_training(self, symbol_count = None, validate: bool = False):
        if not isinstance(self.data_manager.symbols, list):
            print('No symbols loaded')
            return
        if not isinstance(self.ai_manager.model, tf.keras.Model):
            print(f'Creating new model because no model exist: {self.ai_manager.ai_location}')
            self.ai_manager.create_model((self.window, self.features + self.ai_manager.one_hot_encoding_count))
            self.ai_manager.save_model()

        shuffled_list = self.data_manager.symbols.copy()
        random.shuffle(shuffled_list)
        if not isinstance(symbol_count, int):
            symbol_count = len(self.data_manager.symbols)

        if symbol_count >= len(shuffled_list):
            symbol_count = len(shuffled_list) - 1
        
        i = 0
        scaled = None
        train_symbol = None
        test_scaled = None
        test_symbol = None
        x_arr = None
        y_arr = None
        x_test_arr = None
        y_test_arr = None
        while i < symbol_count and i < len(shuffled_list):
            if shuffled_list[i] in self.except_symbols:
                symbol_count += 1
                i += 1
                continue
            if not isinstance(train_symbol, str):
                train_symbol = shuffled_list[i]
                i += 1
            if validate:
                if not isinstance(test_symbol, str):
                    test_symbol = shuffled_list[i]
                try:
                    test_scaled = self.ai_manager.scale_for_ai(data=self.data_manager.get_symbol_data(symbol=test_symbol, interval=self.interval))
                    # test_scaled = self.ai_manager.get_one_hot_encoding(values=test_scaled, one_hot_encoding=self.data_index[test_symbol])
                    x_test_arr, y_test_arr = self.ai_manager.get_xy_arrays(values=test_scaled)
                except Exception as e:
                    print(f'Skipping symbol {shuffled_list[i]}. Reason:')
                    print(e)
                    test_symbol = None
                    symbol_count += 1
                    i += 1
                    continue

            try:
                scaled = self.ai_manager.scale_for_ai(data=self.data_manager.get_symbol_data(symbol=train_symbol, interval=self.interval))
                # scaled = self.ai_manager.get_one_hot_encoding(values=scaled, one_hot_encoding=self.data_index[train_symbol])
                x_arr, y_arr = self.ai_manager.get_xy_arrays(values=scaled)
            except Exception as e:
                print(f'Skipping symbol {shuffled_list[i]}. Reason:')
                print(e)
                train_symbol = None
                symbol_count += 1
                i += 1
                continue

            if validate:
                print(f'Train:{train_symbol} Test:{test_symbol}')
                self.ai_manager.train_model(x_train=x_arr, y_train=y_arr, log_name=f'Train{train_symbol}_Test{test_symbol}', x_test=x_test_arr, y_test=y_test_arr)
                if isinstance(test_symbol, str):
                    test_symbol = train_symbol[:]
                else:
                    print('Weird error!!!')
                train_symbol = None
            else:
                print(f'Train:{train_symbol}')
                train_symbol = None
                self.ai_manager.train_model(x_train=x_arr, y_train=y_arr, log_name=f'Train{train_symbol}')
            self.ai_manager.save_model()
            # i += 1