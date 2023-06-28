import datetime
import io
import os
import time
import pandas as pd
import yfinance

from tradinga.settings import DATA_DIR, SYMBOL_FILE, STOCK_DIR


class DataManager:
    symbols = []
    columns_to_drop = ['adj close']
    rsi_window = 14
    macd_largest_window = 26

    # Indicators
    use_rsi = False
    use_macd = False

    # Data variety
    apply_60m = False
    apply_30m = False

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
        Makes a copy of data to not modify original values and gets time period (including).

        Args:
            data (pd.DataFrame): Preprocessed data.
            date_from (datetime.datetime): Date from.
            date_to (datetime.datetime): Date to.

        Returns:
            Filtered data.
        """
        filtered_data = data.copy()
        if isinstance(date_from, datetime.datetime):
            filtered_data = filtered_data[(filtered_data["time"].dt.date >= date_from.date())]
        if isinstance(date_to, datetime.datetime):
            filtered_data = filtered_data[(filtered_data["time"].dt.date <= date_to.date())]
        return filtered_data

    def yfinance_get_data(
        self, symbol: str, interval: str, max_tries=2, try_sleep=5, show_symbol: bool=False
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
        if interval in ["60m", '1h']:
            from_date = datetime.datetime.now() - datetime.timedelta(days=730)
        elif interval in ["1m", "2m", "5m", "90m", "30m", "15m"]:
            from_date = datetime.datetime.now() - datetime.timedelta(days=60)
        else:
            from_date = datetime.datetime.now() - datetime.timedelta(days=3000)

        if show_symbol:
            print(f"Downloading data for {symbol}")
        tries = 0
        while True:
            try:
                tries += 1
                data = yfinance.download(
                    symbol, start=from_date, interval=interval, progress=False, timeout=1
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

        if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", '1h']:
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
        if len(data.columns) != 7 or 'time' not in data.columns or 'open' not in data.columns or 'close' not in data.columns or 'high' not in data.columns or 'low' not in data.columns or 'adj close' not in data.columns or 'volume' not in data.columns:
                raise Exception(f'Save data called on modified data: {data.columns}')
        try:
            # Discard values for today because it is not final ones during active day
            filtered_data = data[(data["time"].dt.date < datetime.date.today())]

            existing_data = pd.read_csv(file_name)
            existing_data["time"] = pd.to_datetime(
                existing_data["time"], format="ISO8601"
            )
            concatenated_data = pd.concat([filtered_data, existing_data])
            new_data = concatenated_data.drop_duplicates(subset="time")
            new_data = new_data.sort_values(by="time")
            new_data.to_csv(file_name, index=False)
        except:
            data.to_csv(file_name, index=False)

    def get_symbol_data(self, symbol: str, interval: str, online=False, additional_data: bool = True):
        """
        Gets symbol data.
        If file not available locally, then it will be downloaded.

        Args:
            symbol (str): Market symbol.
            interval (str): Market interval.
            online (bool): Force to download new data.

        Returns:
            Data

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

        for to_drop in self.columns_to_drop:
            data = data.drop(to_drop, axis=1)

        if self.use_rsi:
            self.apply_rsi(data=data)
        if self.use_macd:
            self.apply_macd(data=data)

        # Cut off Nan values
        cut_off = 0
        if self.use_rsi:
            cut_off = self.rsi_window
        if self.use_macd:
            if cut_off < self.macd_largest_window:
                cut_off = self.macd_largest_window
        if cut_off > 0:
            data = data[cut_off:]

        # Move column 'close' to the start
        cols = data.columns.tolist()
        cols = ['close'] + [col for col in cols if col != 'close']
        data = data[cols]

        # TODO: FIX WRONG IMPLEMENTATION
        # if self.apply_60m and additional_data:
        #     data = self.add_additional_interval(data=data, symbol=symbol, interval='60m', online=online)
        #     if 'close_60m' not in data.columns:
        #         raise Exception('Could not get 60m data!')
        # if self.apply_30m and additional_data:
        #     data = self.add_additional_interval(data=data, symbol=symbol, interval='30m', online=online)
        #     # print(data.columns)
        #     # print(data)
        #     if 'close_30m' not in data.columns:
        #         raise Exception('Could not get 30m data!')

        return data
    
    def apply_rsi(self, data: pd.DataFrame):
        """
        Applies RSI metric on data. Including NaN values if there are some

        Args:
            data (pd.DataFrame): data

        """
        delta = data["close"].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        avg_gain = gain.rolling(window=self.rsi_window).mean()
        avg_loss = loss.rolling(window=self.rsi_window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        data['rsi'] = rsi

    def apply_macd(self, data: pd.DataFrame):
        """
        Applies MACD metric on data. Including NaN values if there are some

        Args:
            data (pd.DataFrame): data

        """
        exp12 = data["close"].ewm(span=12, adjust=False).mean()
        exp26 = data["close"].ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        data['macd_histogram'] = histogram
        data['macd'] = macd
        data['macd_signal'] = signal

    def add_additional_interval(self, data: pd.DataFrame, symbol: str, interval: str, online=False):
        """
        Adds additional time interval data at the same length

        Args:
            data (pd.DataFrame): data
            symbol (str): Market symbol.
            interval (str): Market interval.
            online (bool): Force to download new data.

        """
        new_data = self.get_symbol_data(symbol=symbol, interval=interval, online=online, additional_data=False)
        if max(data["time"]).date() != max(data["time"]).date():
            raise Exception(f'Data max time {max(data["time"]).date()} and hourly data max time {max(data["time"]).date()} unsync')
        new_data = new_data.drop("time", axis=1)
        if len(data) > len(new_data):
            data = data[-len(new_data):].reset_index(drop=True)
        else:
            new_data = new_data[-len(data):].reset_index(drop=True)

        for col in new_data.columns:
            if col in self.columns_to_drop:
                continue
            data[f'{col}_{interval}'] = new_data[col]
        return data
        

    def get_feature_count(self) -> int:
        """
        Gets feature count for each time unit

        """
        features = 6 - len(self.columns_to_drop)
        if self.apply_60m:
            features += 6 - len(self.columns_to_drop)
        if self.use_rsi:
            features += 1
        if self.use_macd:
            features += 3
        
        return features