



import datetime
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from tradinga import settings
from tradinga.data_analyzer import DataAnalyzer
from tradinga.tools import bcolors


class BusinessLogic:
    """
    Class to realize all user need functionality.
    """

    min_max = True
    current_date = datetime.datetime.now()
    def __init__(self) -> None:
        self.data_analyzer = DataAnalyzer(window=200)
        self.data_analyzer.ai_manager.load_model()

    def get_predicted_change(self, symbol: str, last_date: datetime.datetime) -> tuple[datetime.date, tuple[float, float, float, float, float, float, float]]:
        """
        Gets last real value predicted price change in percentage and value. Based on last_real -> predicted and last_predicted -> predicted changes
        Args:
            symbol (str): Symbol name.

        Returns:
            tuple[last real, real percentage change, real value change, last predicted, predicted percentage change, predicted value change, trend precision]
        """
        result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        data = self.data_analyzer.data_manager.get_symbol_data(symbol=symbol, interval=self.data_analyzer.interval)#, online=True)
        if max(data['time']) < last_date:
            data = self.data_analyzer.data_manager.get_symbol_data(symbol=symbol, interval=self.data_analyzer.interval, online=True)
        filtered_data = self.data_analyzer.data_manager.filter_data(data=data, date_to=last_date)
        scaled_data = self.data_analyzer.ai_manager.scale_for_ai(filtered_data)

        predicted = self.data_analyzer.ai_manager.scale_back_value(self.data_analyzer.ai_manager.predict_next_value(values=scaled_data))[0]

        # last_real -> predicted
        result[0] = filtered_data.iloc[-1]['close']
        result[1] = round((predicted-result[0])/result[0]*100, 2)
        result[2] = round(predicted - result[0], 2)
        # last_predicted -> predicted
        result[3] = self.data_analyzer.ai_manager.scale_back_value(self.data_analyzer.ai_manager.predict_next_value(values=scaled_data[:-1]))[0]
        result[4] = round((predicted-result[3])/result[3]*100, 2)
        result[5] = round(predicted - result[3], 2)
        result[6] = self.data_analyzer.ai_manager.get_metrics_on_data(values=scaled_data, symbol=symbol)[0]
        
        return (datetime.datetime.date(filtered_data.iloc[-1]["time"]), tuple(result))

    def predict_next_close(self, symbol: str, last_date = current_date):
        """
        Shows next possible price value and change for stock. Shows data information.
        Args:
            symbol (str): Symbol name.
        """
        
        result_date, result = self.get_predicted_change(symbol=symbol, last_date=last_date)
        if result_date != last_date.date():
                print(f'{bcolors.WARNING}Last date: {result_date} of {symbol} is not correct!{bcolors.ENDC}')
        if self.data_analyzer.in_exclude(symbol=symbol):
            print(f'{bcolors.WARNING}Warning! Symbol is in except symbol list{bcolors.ENDC}')
        print(f'For stock {symbol} after: {result_date} with price: {result[0]}')
        print(f'Last real -> Predicted change: {result[1]}% that is: {result[2]}')
        print(f'Last predicted -> Predicted change: {result[4]}% that is: {result[5]}')

    def show_prediction_chart(self, symbol: str):
        """
        Shows prediction chart to see if predictions are not too off the chart.
        Args:
            symbol (str): Symbol name.
        """
        data = self.data_analyzer.data_manager.get_symbol_data(symbol=symbol, interval=self.data_analyzer.interval)
        scaled_data = self.data_analyzer.ai_manager.scale_for_ai(data)
        predicted = self.data_analyzer.ai_manager.scale_back_value(self.data_analyzer.ai_manager.predict_all_values(values=scaled_data))
        plot_data = data[['time', 'close']].copy()
        # plot_data = plot_data.iloc[self.data_analyzer.window + self.data_analyzer.ai_manager.predict_after_time:]
        # plot_data['predicted'] = predicted[:len(predicted)-self.data_analyzer.ai_manager.predict_after_time]
        plot_data = plot_data.iloc[self.data_analyzer.window:]
        plot_data['predicted'] = predicted
        if self.data_analyzer.in_exclude(symbol=symbol):
            print(f'{bcolors.WARNING}Warning! Symbol is in except symbol list{bcolors.ENDC}')
        plt.plot(plot_data['time'], plot_data['close'], label='Actual')
        plt.plot(plot_data['time'], plot_data['predicted'], label='Predicted')
        plt.legend()
        plt.show()

    def improve_model(self, symbol_count = None):
        """
        Improves existing or creates new model.
        Args:
            symbol_count (int): On how many symbols model has to be trained on.
        """
        if not isinstance(symbol_count, int):
            symbol_count = len(self.data_analyzer.data_index)
        if self.min_max and not self.data_analyzer.settings:
            self.data_analyzer.filter_stocks()
            self.data_analyzer.save_settings()
        self.data_analyzer.random_training(symbol_count=symbol_count, validate=True)

    def print_metric_summary(self, symbol: str):
        """
        Improves existing or creates new model.
        Args:
            symbol_count (int): On how many symbols model has to be trained on.
        """
        data = self.data_analyzer.data_manager.get_symbol_data(symbol=symbol, interval=self.data_analyzer.interval)
        scaled_data = self.data_analyzer.ai_manager.scale_for_ai(data)
        metrics = self.data_analyzer.ai_manager.get_metrics_on_data(values=scaled_data, symbol=symbol)
        print(f'Trend precision for {symbol} is: {metrics[0]}')

    def predict_market(self, last_date: datetime.datetime = datetime.datetime.now()):
        """
        Gets model predictions for stock market and saves data in file.

        Prediction file will consist of columns:
        symbol;
        prediction based prediction (prediction calculated by looking at previous predicted value) percentage/price;
        real value based prediction (prediction calculated by looking at previous real value) percentage/price;
        Args:
            last_date (datetime.datetime): Date to start prediction from.
        """
        columns = ['symbol', 'price', 'predicted_real_change', 'pred_real_percent', 'last_predicted_price', 'predicted_change', 'pred_percent']
        market_results = pd.DataFrame(columns=columns)

        for symbol in tqdm.tqdm(self.data_analyzer.data_manager.symbols):
            # print(f'Processing {symbol}')
            if self.data_analyzer.in_exclude(symbol=symbol):
                continue
            try:
                data_date, results = self.get_predicted_change(symbol=symbol, last_date=last_date)
            except Exception as e:
                # print(f'Reason: {e}')
                continue
            if data_date != last_date.date():
                # print(f'{bcolors.WARNING}Last date: {data_date} of {symbol} is not correct! {bcolors.ENDC}')
                continue

            new_row = pd.DataFrame({
                'symbol': [symbol],
                'price': [results[0]],
                'pred_real_percent': [results[1]],
                'predicted_real_change': [results[2]],
                'last_predicted_price': [results[3]],
                'pred_percent': [results[4]],
                'predicted_change': [results[5]],
                'trend_precision': [results[6]]
            })
            # print(new_row)
            market_results = pd.concat([market_results, new_row], ignore_index=True)
            # print(market_results)

        if not os.path.exists(f'{settings.DATA_DIR}/predictions'):
            os.makedirs(f'{settings.DATA_DIR}/predictions')
        market_results.to_csv(f'{settings.DATA_DIR}/predictions/Predictions_{last_date.date()}.csv', index=False)

