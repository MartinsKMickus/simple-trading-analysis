import datetime
import io
import os
import time
import pandas as pd
import yfinance

from tradinga.settings import DATA_DIR, SYMBOL_FILE, STOCK_DIR


class DataManager:
    symbols = []

    def __init__(self, data_dir: str = DATA_DIR, stock_dir: str = STOCK_DIR) -> None:
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.stock_dir = data_dir + "/" + stock_dir
        if not os.path.exists(self.stock_dir):
            os.makedirs(self.stock_dir)

    def get_nasdaq_symbols(
        self, symbol_file: str = SYMBOL_FILE, online=False
    ) -> pd.DataFrame:
        """
        Gets NASDAQ symbol list and saves it in class variable.
        If file not available locally, then it will be downloaded.

        Args:
            symbol_file (str): Path to csv file where to store symbols.
            online (bool): Force to download new data.

        Returns:
            Symbol data.
        """
        symbol_data = None
        if os.path.exists(symbol_file):
            symbol_data = pd.read_csv(symbol_file)
        if online or not isinstance(symbol_data, pd.DataFrame):
            import requests

            url = "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
            response = requests.get(url)
            symbol_data = pd.read_csv(io.StringIO(response.text), delimiter="|")
            symbol_data.columns = symbol_data.columns.str.lower()
            symbol_data = symbol_data.dropna(subset=["symbol"])
            symbol_data = symbol_data.reset_index(drop=True)
            symbol_data.to_csv(symbol_file, index=False)

        if not isinstance(symbol_data, pd.DataFrame):
            raise Exception("Something went wrong getting NASDAQ symbols")

        self.symbols = symbol_data["symbol"].to_list()
        return symbol_data

    def filter_data(self, data: pd.DataFrame, date_from=None, date_to=None):
        """
        Makes a copy of data to not modify original values and gets time period.

        Args:
            data (pd.DataFrame): Preprocessed data.
            date_from (datetime.datetime): Date from.
            date_to (datetime.datetime): Date to.

        Returns:
            Filtered data.
        """
        filtered_data = data.copy()
        if date_from is not None:
            filtered_data = filtered_data[(filtered_data["time"] > date_from)]
        if date_to is not None:
            filtered_data = filtered_data[(filtered_data["time"] < date_to)]
        return filtered_data

    def yfinance_get_data(
        self, symbol: str, interval: str, max_tries=2, try_sleep=10, show_symbol: bool=False
    ) -> pd.DataFrame:
        """
        Downloads data for provided symbol and interval.

        Args:
            symbol (str): Market symbol.
            interval (str): Market interval.
            max_tries (int): If failed to download, how many times to retry.
            try_sleep (int): How long to wait between tries.

        Returns:
            Downloaded data
        """
        if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
            from_date = datetime.datetime.now() - datetime.timedelta(days=730)
        else:
            from_date = datetime.datetime.now() - datetime.timedelta(days=3000)

        if show_symbol:
            print(f"Downloading data for {symbol}")
        tries = 0
        while True:
            try:
                tries += 1
                data = yfinance.download(
                    symbol, start=from_date, interval=interval, progress=False
                )
                break
            except:
                if tries >= max_tries:
                    raise Exception(f"Failed to download {symbol}")
                print(f"Failed to download {symbol}. Retrying in {try_sleep}s")
                time.sleep(try_sleep)
                continue

        if not isinstance(data, pd.DataFrame):
            raise Exception(f"Failed to download? {symbol}")

        # date/time column is index. Need to make it as normal column.
        data = data.reset_index()

        data = data.dropna(how="any")
        data.columns = data.columns.str.lower()

        if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
            data = data.rename(columns={"datetime": "time"})
        else:
            data = data.rename(columns={"date": "time"})

        data["time"] = pd.to_datetime(data["time"], format="ISO8601")

        return data

    def save_data_to_csv(self, data: pd.DataFrame, file_name: str):
        """
        Saves symbol data to csv. Updates local data if possible.

        Args:
            data (pd.DataFrame): Preprocessed symbol data.
            file_name (str): Path to csv file where to save data.

        """
        try:
            existing_data = pd.read_csv(file_name)
            existing_data["time"] = pd.to_datetime(
                existing_data["time"], format="ISO8601"
            )
            concatenated_data = pd.concat([data, existing_data])
            new_data = concatenated_data.drop_duplicates(subset="time")
            new_data = new_data.sort_values(by="time")
            new_data.to_csv(file_name, index=False)
        except:
            data.to_csv(file_name, index=False)

    def get_symbol_data(self, symbol: str, interval: str, online=False):
        """
        Gets symbol data.
        If file not available locally, then it will be downloaded.

        Args:
            symbol (str): Market symbol.
            interval (str): Market interval.
            online (bool): Force to download new data.

        """
        data = None
        file_path = f"{self.stock_dir}/{symbol}_{interval}.csv"  # os.path.join(STOCK_DIR, f'{symbol}_{interval}.csv')
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)

        if online or not isinstance(data, pd.DataFrame):
            data = self.yfinance_get_data(symbol=symbol, interval=interval)
            self.save_data_to_csv(data=data, file_name=file_path)

        data = data.reset_index(drop=True)
        data["time"] = pd.to_datetime(data["time"], format="ISO8601")
        return data
