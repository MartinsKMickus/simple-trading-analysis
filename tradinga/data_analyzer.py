

import datetime
import json
import os
import random
import sys
from matplotlib import pyplot as plt
import numpy as np

from tradinga.tools import bcolors, query_yes_no
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tqdm
from tradinga.ai_manager import AIManager
from tradinga.data_manager import DataManager
from tradinga.settings import DATA_DIR, MIN_DATA_CHECKS, STOCK_DIR, SYMBOL_FILE


class DataAnalyzer:
    data_index = {}
    fit_times = 0
    precision = 0
    model_except = []
    model_highest_loss = 0.14

    def __init__(self, analyzer_name:str = 'generic_analyzer', data_dir: str = DATA_DIR, stock_dir: str = STOCK_DIR, symbol_file: str = SYMBOL_FILE, window: int = 200) -> None:
        # Settings
        self.analyzer_name = analyzer_name
        self.data_dir = data_dir

        # Additional settings
        self.min_values = [0, 0, 0, 0, 0, 0]
        self.max_values = [700, 700, 700, 700, 700, 1000000]
        self.use_min_max = True
        self.except_symbols = []
        self.load_settings()
        self.interval = '1d'
        self.window = window
        self.features = 6
        self.min_data_checks = MIN_DATA_CHECKS
        self.data_manager = DataManager(data_dir=data_dir, stock_dir=stock_dir)
        self.data_manager.get_nasdaq_symbols(symbol_file=symbol_file)
        # One hot encoding
        self.one_hot = len(self.data_manager.symbols)
        self.use_one_hot = False

        # AI Manager
        # One hot encoding included
        if self.settings:
            if self.use_min_max:
                # print(f'Settings loaded!\n Applied min: {self.min_values}\n Applied max: {self.max_values}')
                self.ai_manager = AIManager(data_dir=data_dir, model_name=self.analyzer_name, data_min=np.array(self.min_values), data_max=np.array(self.max_values), one_hot_encoding_count=self.one_hot)
            else:
                self.ai_manager = AIManager(data_dir=data_dir, model_name=self.analyzer_name, one_hot_encoding_count=self.one_hot)
        else:
            print(f'WARNING! Settings not loaded!')
            self.ai_manager = AIManager(data_dir=data_dir, model_name=self.analyzer_name, one_hot_encoding_count=self.one_hot)
        self.ai_manager.window = self.window
        self.ai_manager.data_columns = self.features

        # Last initializations
        self.load_symbol_indices()

    def load_settings(self):
        """
        Loads settings stored locally. If settings loaded then self.settings is set to True, else False

        """
        if os.path.exists(f'{self.data_dir}/{self.analyzer_name}_settings.json'):
            with open(f'{self.data_dir}/{self.analyzer_name}_settings.json', "r") as file:
                try:
                    loaded_data = json.load(file)
                    self.use_min_max, self.min_values, self.max_values, self.except_symbols, self.fit_times, self.precision, self.model_except, self.model_highest_loss = loaded_data
                    print('Settings loaded!')
                    if self.use_min_max:
                        print(f'Applied min: {self.min_values}\nApplied max: {self.max_values}')
                    print(f'Fit times: {self.fit_times}\nPrecision: {self.precision}')
                    print(f'Highest loss: {self.model_highest_loss}')
                except Exception:
                    print(f'{bcolors.FAIL}Corrupted settings file {self.data_dir}/{self.analyzer_name}_settings.json{bcolors.ENDC}')
                    if query_yes_no('Delete corrupted file?', default='no'):
                        os.remove(f'{self.data_dir}/{self.analyzer_name}_settings.json')
                        self.settings = False
                    else:
                        sys.exit(0)
            self.settings = True
        self.settings = False
    
    def save_settings(self, overwrite: bool = False):
        """
        Saves settings in local file.

        Args:
            overwrite (bool): Overrwrite existing file
        """
        if os.path.exists(f'{self.data_dir}/{self.analyzer_name}_settings.json'):
            if not overwrite:
                print(f'Settings already saved. Use overwrite mode.')
                return
        with open(f'{self.data_dir}/{self.analyzer_name}_settings.json', 'w') as file:
            settings_definition = (self.use_min_max, self.min_values, self.max_values, self.except_symbols, self.fit_times, self.precision, self.model_except, self.model_highest_loss)
            json.dump(settings_definition, file)

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
        """
        Filters out stocks which values exceeds min max scaler interval.
        Probably reasonable price range would be somewhere around 0-500
        Volume can't be predicted that easy because of events like 2008

        """
        if os.path.exists(f'{self.data_dir}/{self.analyzer_name}_settings.json') and not force:
            print('Settings already stored')
        if not isinstance(self.data_manager.symbols, list):
            print('No symbols loaded')
            return
        
        for symbol in tqdm.tqdm(self.data_manager.symbols, desc='Filtering out stocks'):#self.data_manager.symbols:
            loaded_data = self.data_manager.get_symbol_data(symbol=symbol, interval=self.interval)

            value = loaded_data['open'].max()
            if value > self.max_values[0]:
                if value > 500:
                    # print(f'{symbol} price is too high: {value}')
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
        
        data_min = [float(item) for item in self.min_values]
        data_max = [float(item) for item in self.max_values]
        self.min_values = data_min
        self.max_values = data_max

        print(f'Filtered out {len(self.except_symbols)} symbols out of {len(self.data_manager.symbols)}')
        print(f'Applied min: {self.min_values}\nApplied max: {self.max_values}')
        # Save variables to a file
        # with open(f'{self.data_dir}/{self.analyzer_name}_settings.json', 'w') as file: !!!!!!!!!!!!! DON'T USE
        #     json.dump((self.min_values, self.max_values, self.except_symbols), file)

    def random_valuation(self, symbol_count = 50):
        """
        Gets valuation metrics for different stocks.
        
        Args:
            symbol_count (int): How much symbols should be analyzed to get metrics.
        Returns:
            List with lists of symbols and metric related to it. [[symbol, metric], ...]
        """
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
            # One hot encoding
            if self.use_one_hot:
                one_hot_encoding = self.ai_manager.get_one_hot_encoding(one_hot_encoding=self.data_index[shuffled_list[i]])
                metrics.append([shuffled_list[i], self.ai_manager.get_metrics_on_data(scaled, symbol=shuffled_list[i], one_hot_encoding=one_hot_encoding)])
            else:
                metrics.append([shuffled_list[i], self.ai_manager.get_metrics_on_data(scaled, symbol=shuffled_list[i])])
            i += 1
        return metrics

    def random_training(self, symbol_count = None, validate: bool = False):
        """
        Heart of "Simple Trading Analysis". Trains model on data and does some of the data filtering, updates settings and model file.
        
        Args:
            symbol_count (int): How much symbols should be analyzed to get metrics. (Default: None (all))
        Returns:
            List with lists of symbols and metric related to it. [[symbol, metric], ...]
        """
        if not isinstance(self.data_manager.symbols, list):
            print('No symbols loaded')
            return
        if not isinstance(self.ai_manager.model, tf.keras.Model):
            print(f'Creating new model because no model exist: {self.ai_manager.ai_location}')

            # One hot encoding
            if self.use_one_hot:

                self.ai_manager.create_model2(data_shape=(self.window, self.features), category_shape=(self.one_hot))
            else:
                self.ai_manager.create_model((self.window, self.features))
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
        # One hot encoding
        one_hot_train = None
        one_hot_test = None

        while i < symbol_count and i < len(shuffled_list):
            if shuffled_list[i] in self.except_symbols or shuffled_list[i] in self.model_except:
                # print(f'Skipping symbol {shuffled_list[i]}. Symbol in except list')
                symbol_count += 1
                i += 1
                continue
            try:
                prefiltering = self.ai_manager.scale_for_ai(data=self.data_manager.get_symbol_data(symbol=shuffled_list[i], interval=self.interval))
                if len(prefiltering) < self.window + MIN_DATA_CHECKS:
                    raise Exception(f'Not enough values {len(prefiltering)}')
                # One hot encoding
                if self.use_one_hot:
                    prefiltering_one_hot = self.ai_manager.get_one_hot_encoding(one_hot_encoding=self.data_index[test_symbol])
                x_prefilter, y_prefilter = self.ai_manager.get_xy_arrays(values=prefiltering)
            except Exception as e:
                print(f'Skipping symbol {shuffled_list[i]}. Reason:', end=' ')
                print(e)
                test_symbol = None
                symbol_count += 1
                i += 1
                continue
            # Optimizer
            if self.model_highest_loss < self.ai_manager.get_evaluation(x_array=x_prefilter, y_array=y_prefilter)[0] and self.fit_times > 1:
                # print(f'Skipping symbol {shuffled_list[i]}. Symbol evaluation value too low')
                self.model_except.append(shuffled_list[i])
                self.save_settings(overwrite=True)
                symbol_count += 1
                i += 1
                continue
            if not isinstance(train_symbol, str):
                train_symbol = shuffled_list[i]
                i += 1
                continue
            if validate:
                if not isinstance(test_symbol, str):
                    test_symbol = shuffled_list[i]
                try:
                    test_scaled = self.ai_manager.scale_for_ai(data=self.data_manager.get_symbol_data(symbol=test_symbol, interval=self.interval))
                    # One hot encoding
                    if self.use_one_hot:
                        one_hot_test = self.ai_manager.get_one_hot_encoding(one_hot_encoding=self.data_index[test_symbol])
                    x_test_arr, y_test_arr = self.ai_manager.get_xy_arrays(values=test_scaled)
                except Exception as e:
                    print(f'Skipping symbol {shuffled_list[i]}. Reason:', end=' ')
                    print(e)
                    test_symbol = None
                    symbol_count += 1
                    i += 1
                    continue

            try:
                scaled = self.ai_manager.scale_for_ai(data=self.data_manager.get_symbol_data(symbol=train_symbol, interval=self.interval))
                # One hot encoding
                if self.use_one_hot:
                    one_hot_train = self.ai_manager.get_one_hot_encoding(one_hot_encoding=self.data_index[train_symbol])
                x_arr, y_arr = self.ai_manager.get_xy_arrays(values=scaled)
            except Exception as e:
                print(f'Skipping symbol {shuffled_list[i]}. Reason:', end=' ')
                print(e)
                train_symbol = None
                symbol_count += 1
                i += 1
                continue

            try:
                if validate:
                    print(f'Train:{train_symbol} Test:{test_symbol}')
                    # One hot encoding
                    if self.use_one_hot:
                        self.ai_manager.train_model(x_train=x_arr, one_hot=one_hot_train, y_train=y_arr, log_name=f'Train{train_symbol}_Test{test_symbol}', x_test=x_test_arr, one_hot_test=one_hot_test, y_test=y_test_arr)
                    else:
                        self.ai_manager.train_model(x_train=x_arr, y_train=y_arr, log_name=f'Train{train_symbol}_Test{test_symbol}', x_test=x_test_arr, y_test=y_test_arr)
                    if isinstance(test_symbol, str):
                        test_symbol = train_symbol[:]
                    else:
                        print('Weird error!!!')
                else:
                    print(f'Train:{train_symbol}')
                    
                    # One hot encoding
                    if self.use_one_hot:
                        self.ai_manager.train_model(x_train=x_arr, one_hot=one_hot_train, y_train=y_arr, log_name=f'Train{train_symbol}')
                    else:
                        self.ai_manager.train_model(x_train=x_arr, y_train=y_arr, log_name=f'Train{train_symbol}')
            except KeyboardInterrupt:
                print("Ctrl+C detected. Stopping the program...")
                # Perform any necessary cleanup or finalization steps
                sys.exit(0)

            self.fit_times += 1
            if isinstance(test_symbol, str):
                new_precision = (self.precision * (self.fit_times - 1) + self.ai_manager.get_metrics_on_data(values=scaled, symbol=test_symbol)[0]) / self.fit_times
            else:
                new_precision = (self.precision * (self.fit_times - 1) + self.ai_manager.get_metrics_on_data(values=scaled, symbol=train_symbol)[0]) / self.fit_times
            if new_precision > self.precision:
                print(f"{bcolors.OKGREEN}Great! Average precision increased: {round(self.precision, 5)} -> {round(new_precision, 5)}{bcolors.ENDC}")
            elif new_precision == self.precision:
                print(f"{bcolors.CBLINK}Average precision stayed the same: {round(self.precision, 5)}{bcolors.ENDC}")
            else:
                print(f"{bcolors.WARNING}Warning! Average precision decreased: {round(self.precision, 5)} -> {round(new_precision, 5)}{bcolors.ENDC}")
            self.precision = new_precision
            fast_eval = self.ai_manager.get_evaluation(x_array=x_prefilter, y_array=y_prefilter)[0]
            # Optimizer every 5 data sets
            if self.model_highest_loss > fast_eval * 700 and self.fit_times % 5 == 0:
                print(f"{bcolors.CBLINK}{bcolors.OKCYAN}Highest allowed loss updated: {self.model_highest_loss} -> {fast_eval * 700}{bcolors.ENDC}")
                self.model_highest_loss = fast_eval * 700
            self.ai_manager.save_model()
            self.save_settings(overwrite=True)
            # i += 1
            train_symbol = None