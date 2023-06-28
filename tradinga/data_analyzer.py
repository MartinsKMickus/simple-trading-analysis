

import datetime
import json
from multiprocessing import Process, Queue
import os
import shutil
import threading
from memory_profiler import profile
import random
import sys
from matplotlib import pyplot as plt
import numpy as np
from tradinga import settings

from tradinga.tools import bcolors, query_yes_no
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tqdm
from tradinga.ai_manager import AIManager
from tradinga.data_manager import DataManager
from tradinga.settings import DATA_DIR, MIN_DATA_CHECKS, STOCK_DIR, SYMBOL_FILE

# @profile
class DataAnalyzer:

    def __init__(self, analyzer_name:str = 'generic_analyzer', data_dir: str = DATA_DIR, stock_dir: str = STOCK_DIR, symbol_file: str = SYMBOL_FILE, window: int = 100) -> None:
        # Settings
        self.analyzer_name = analyzer_name
        self.data_dir = data_dir

        # Additional settings
        self.data_index = {}
        self.fit_times = 0
        self.precision = 0
        self.model_except = []
        self.model_lowest_precision = 0
        self.min_values = [0, 0, 0, 0, 0, 0]
        self.max_values = [700, 700, 700, 700, 700, 1000000]
        self.use_min_max = False
        self.except_symbols = []
        self.load_settings()
        self.interval = settings.INTERVAL
        self.window = window
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
                self.ai_manager = AIManager(data_dir=data_dir, model_name=self.analyzer_name, data_min=np.array(self.min_values), data_max=np.array(self.max_values), one_hot_encoding_count=self.one_hot, window=self.window)
            else:
                self.ai_manager = AIManager(data_dir=data_dir, model_name=self.analyzer_name, one_hot_encoding_count=self.one_hot, window=self.window)
        else:
            print(f'WARNING! Settings not loaded!')
            self.ai_manager = AIManager(data_dir=data_dir, model_name=self.analyzer_name, one_hot_encoding_count=self.one_hot, window=self.window)
        self.ai_manager.window = self.window
        self.ai_manager.data_columns = self.data_manager.get_feature_count()

        # Last initializations
        self.load_symbol_indices()
        self.ai_manager.load_model()

    def load_settings(self):
        """
        Loads settings stored locally. If settings loaded then self.settings is set to True, else False

        """
        if os.path.exists(f'{self.data_dir}/{self.analyzer_name}_settings.json'):
            with open(f'{self.data_dir}/{self.analyzer_name}_settings.json', "r") as file:
                try:
                    loaded_data = json.load(file)
                    self.use_min_max, self.min_values, self.max_values, self.except_symbols, self.fit_times, self.precision, self.model_except, self.model_lowest_precision = loaded_data
                    print('Settings loaded!')
                    if self.use_min_max:
                        print(f'Applied min: {self.min_values}\nApplied max: {self.max_values}')
                    print(f'Fit times: {self.fit_times}')
                    print(f'Average precision: {self.precision}')
                    print(f'Lowest allowed precision: {self.model_lowest_precision}')
                except Exception:
                    print(f'{bcolors.FAIL}Corrupted settings file {self.data_dir}/{self.analyzer_name}_settings.json{bcolors.ENDC}')
                    if query_yes_no('Delete corrupted file?', default='no'):
                        os.remove(f'{self.data_dir}/{self.analyzer_name}_settings.json')
                        self.settings = False
                        return
                    else:
                        sys.exit(0)
            self.settings = True
            return
        self.settings = False
        return
    
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
            settings_definition = (self.use_min_max, self.min_values, self.max_values, self.except_symbols, self.fit_times, self.precision, self.model_except, self.model_lowest_precision)
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

    def reset_model_and_settings(self):
        """
        Deletes model file and resets settings except filter

        """
        if query_yes_no(question="Are you sure to delete model and reset settings?", default='no'):
            if os.path.exists(self.ai_manager.ai_location):
                shutil.rmtree(self.ai_manager.ai_location, ignore_errors=True)
            self.ai_manager.create_model((self.window, self.data_manager.get_feature_count()))
            self.fit_times = 0
            self.model_except = []
            self.precision = 0
            self.model_lowest_precision = 0
            self.save_settings(overwrite=True)

    def filter_stocks(self, force = False, apply_scaler = False):
        """
        Filters out stocks which values exceeds min max scaler interval.
        Probably reasonable price range would be somewhere around 0-500
        Volume can't be predicted that easy because of events like 2008

        Applies filter for ai_manager

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
        if apply_scaler:
            self.ai_manager.apply_minmax_setting(data_min=np.array(self.min_values), data_max=np.array(self.max_values))
            print(f'Applied min: {self.min_values}\nApplied max: {self.max_values}')
        # Save variables to a file
        # with open(f'{self.data_dir}/{self.analyzer_name}_settings.json', 'w') as file: !!!!!!!!!!!!! DON'T USE
        #     json.dump((self.min_values, self.max_values, self.except_symbols), file)

    def in_exclude(self, symbol: str) -> bool:
        """
        Checks whatever symbol is or isn't in exclude symbol list.
        Args:
            symbol (str): Symbol.
        Returns:
            True if symbol is in exclude list and False if it isn't (bool)
        """
        if symbol in self.except_symbols or symbol in self.model_except:
            return True
        return False

    def get_valuation(self, symbol:str):
        """
        Gets valuation metrics for specific stock.
        
        Args:
            symbol (str): Symbol.
        Returns:
            List with lists of symbol name at start and metrics related to it. [symbol, metrics]
        """
        loaded_data = self.data_manager.get_symbol_data(symbol=symbol, interval=self.interval)
        if len(loaded_data) < self.window + self.min_data_checks:
            print(f'{bcolors.FAIL}Symbol {symbol} has not enough data points{bcolors.ENDC}')
            return [symbol, 'Not enough data points']
        scaled = self.ai_manager.scale_for_ai(data=loaded_data)
        return [symbol, self.ai_manager.get_metrics_on_data(scaled, symbol=symbol)]

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

    def update_plot(self, queue: Queue):
        """
        Plot function to display some info within process loop. To not freeze plot. should be ended with terminate.
        
        Args:
            queue (Queue): Queue for plotting. Info should be added later with: queue.put(array). Info provided: (accuracy_plot, validation_symbol, validation_value, validation_plot)
        
        """
        plt.ion()
        # _, ax = plt.subplots()
        fig, (ax1, ax2) = plt.subplots(1, 2)  # Create two subplots side by side
        fig.tight_layout()  # Adjust spacing between subplots
        while True:
            if not queue.empty():
                data = queue.get()
                ax1.clear()
                ax1.plot(data[0])
                ax1.set_title('Trend way prediction accuracy')
                ax2.clear()
                ax2.plot(data[3]['time'], data[3]['close'], label='Actual')
                ax2.plot(data[3]['time'], data[3]['predicted'], label='Predicted')
                ax2.set_title(f'Validation on {data[1]}: {data[2]}')
                # ax.clear()
                # ax.plot(queue.get())
                plt.draw()
            plt.pause(0.1)
        # plt.close('all')

    def random_training(self, symbol_count = None, validate: bool = False, wait_end: bool = True):
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

                self.ai_manager.create_model2(data_shape=(self.window, self.data_manager.get_feature_count()), category_shape=(self.one_hot))
            else:
                self.ai_manager.create_model((self.window, self.data_manager.get_feature_count()))
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

        # Statistics
        self.precision_values = []
        queue = Queue()
        plot_process = Process(target=self.update_plot, args=(queue,))
        plot_process.start()
        data = self.data_manager.get_symbol_data(symbol='AAPL', interval=self.interval)
        scaled = self.ai_manager.scale_for_ai(data=data)
        val_metric = self.ai_manager.get_metrics_on_data(values=scaled, symbol='AAPL')[0]
        predicted = self.ai_manager.scale_back_value(self.ai_manager.predict_all_values(values=scaled))
        plot_data = data[['time', 'close']].copy()
        plot_data = plot_data.iloc[self.window:]
        plot_data['predicted'] = predicted
        queue.put((self.precision_values, 'AAPL', val_metric, plot_data))

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
                self.model_except.append(shuffled_list[i])
                self.save_settings(overwrite=True)
                print(e)
                test_symbol = None
                symbol_count += 1
                i += 1
                continue
            # Optimizer OLD
            # if self.model_highest_loss < self.ai_manager.get_evaluation(x_array=x_prefilter, y_array=y_prefilter)[0] and self.fit_times > 20:
            #     print(f'Skipping symbol {shuffled_list[i]}. Symbol evaluation results too bad')
            #     self.model_except.append(shuffled_list[i])
            #     self.save_settings(overwrite=True)
            #     symbol_count += 1
            #     i += 1
            #     continue
            # Optimizer NEW
            precision = self.ai_manager.get_metrics_on_data(values=prefiltering, symbol=shuffled_list[i])[0]
            if self.model_lowest_precision > precision:
                print(f'Skipping symbol {shuffled_list[i]}. Symbol precision value too low {round(precision*100, 2)}% < {round(self.model_lowest_precision* 100, 2)}%')
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
                plot_process.terminate()
                sys.exit(0)

            self.fit_times += 1
            if isinstance(test_symbol, str):
                precision = self.ai_manager.get_metrics_on_data(values=scaled, symbol=test_symbol)[0]
            else:
                precision = self.ai_manager.get_metrics_on_data(values=scaled, symbol=train_symbol)[0]
            new_precision = (self.precision * (self.fit_times - 1) + precision) / self.fit_times
            
            # If precision is too low, then model won't be updated ISN'T TRIGGERED
            # if new_precision < self.precision * required_precision_coef:
            #     print(f'Skipping model update because precision is {bcolors.WARNING}{new_precision} < {required_precision_coef * self.precision}{bcolors.ENDC}')
            #     self.ai_manager.load_model()
            #     train_symbol = None
            #     continue

            if new_precision > self.precision:
                print(f"{bcolors.OKGREEN}Great! Average precision increased: {round(self.precision, 5)} -> {round(new_precision, 5)}{bcolors.ENDC}")
            elif new_precision == self.precision:
                print(f"{bcolors.CBLINK}Average precision stayed the same: {round(self.precision, 5)}{bcolors.ENDC}")
            else:
                print(f"{bcolors.WARNING}Warning! Average precision decreased: {round(self.precision, 5)} -> {round(new_precision, 5)}{bcolors.ENDC}")
            self.precision = new_precision

            # Optimizer every 5 data sets DISABLED FOR NOW
            # fast_eval = self.ai_manager.get_evaluation(x_array=x_prefilter, y_array=y_prefilter)[0]
            # if self.model_highest_loss > fast_eval * 100: # and self.fit_times % 5 == 0
            #     print(f"{bcolors.CBLINK}{bcolors.OKCYAN}Highest allowed loss updated: {self.model_highest_loss} -> {fast_eval * 100}{bcolors.ENDC}")
            #     self.model_highest_loss = fast_eval * 100
            # ALTERNATIVE
            required_precision_coef = 0.85
            if self.model_lowest_precision < precision * required_precision_coef and self.fit_times > 0: # and self.fit_times % 5 == 0
                print(f"{bcolors.CBLINK}{bcolors.OKCYAN}Lowest allowed precision updated: {self.model_lowest_precision} -> {precision * required_precision_coef}{bcolors.ENDC}")
                self.model_lowest_precision = precision * required_precision_coef

            # Statistics
            self.precision_values.append(new_precision)
            data = self.data_manager.get_symbol_data(symbol='AAPL', interval=self.interval)
            scaled = self.ai_manager.scale_for_ai(data=data)
            val_metric = self.ai_manager.get_metrics_on_data(values=scaled, symbol='AAPL')[0]
            predicted = self.ai_manager.scale_back_value(self.ai_manager.predict_all_values(values=scaled))
            plot_data = data[['time', 'close']].copy()
            plot_data = plot_data.iloc[self.window:]
            plot_data['predicted'] = predicted
            queue.put((self.precision_values, 'AAPL', val_metric, plot_data))

            # Status
            if symbol_count > len(shuffled_list):
                temp_count = len(shuffled_list) - 1
            else:
                temp_count = symbol_count
            print(f'Random traing progress: {i}/{temp_count} = {round(i/temp_count*100, 2)}%')

            self.ai_manager.save_model()
            self.save_settings(overwrite=True)
            train_symbol = None

        if wait_end:
            input("Press Enter to end...")
        plot_process.terminate()