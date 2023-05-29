

import json
import os
from matplotlib import pyplot as plt
from tradinga.data_manager import DataManager
from tradinga.settings import DATA_DIR, STOCK_DIR


class DataAnalyzer:
    data_index = {}

    def __init__(self, analyzer_name:str = 'generic_analyzer', data_dir: str = DATA_DIR, stock_dir: str = STOCK_DIR) -> None:
        self.data_manager = DataManager(data_dir=data_dir, stock_dir=stock_dir)
        self.data_manager.get_nasdaq_symbols()
        self.analyzer_name = analyzer_name
        self.data_dir = data_dir

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
        