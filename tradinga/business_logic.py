



import datetime

from matplotlib import pyplot as plt
import numpy as np
from tradinga.data_analyzer import DataAnalyzer


class BusinessLogic:
    """
    Class to realize all user need functionality.
    """

    def __init__(self) -> None:
        self.data_analyzer = DataAnalyzer()
        self.data_analyzer.ai_manager.load_model()
        self.current_date = datetime.datetime.now()

    def predict_next_close(self, symbol: str):
        """
        Shows next possible price value and change for stock. Shows data information.
        """
        data = self.data_analyzer.data_manager.get_symbol_data(symbol=symbol, interval=self.data_analyzer.interval)#, online=True)
        filtered_data = self.data_analyzer.data_manager.filter_data(data=data, date_to=self.current_date)
        scaled_data = self.data_analyzer.ai_manager.scale_for_ai(filtered_data)
        predicted = self.data_analyzer.ai_manager.scale_back_value(self.data_analyzer.ai_manager.predict_next_value(values=scaled_data))[0]
        change = (predicted-filtered_data.iloc[-1]['close'])/filtered_data.iloc[-1]['close']*100
        print(f'For stock {symbol} after: {datetime.datetime.date(filtered_data.iloc[-1]["time"])} with price: {round(filtered_data.iloc[-1]["close"], 2)}')
        print(f'Predicted change: {round(change, 2)}% and price: {round(predicted, 2)}')

    def show_prediction_chart(self, symbol: str):
        """
        Shows prediction chart to see if predictions are not too off the chart
        """
        data = self.data_analyzer.data_manager.get_symbol_data(symbol=symbol, interval=self.data_analyzer.interval)
        scaled_data = self.data_analyzer.ai_manager.scale_for_ai(data)
        predicted = self.data_analyzer.ai_manager.scale_back_value(self.data_analyzer.ai_manager.predict_all_values(values=scaled_data))
        plot_data = data[['time', 'close']].copy()

        plot_data = plot_data.iloc[self.data_analyzer.window:]
        plot_data['predicted'] = predicted

        plt.plot(plot_data['time'], plot_data['close'], label='Actual')
        plt.plot(plot_data['time'], plot_data['predicted'], label='Predicted')
        plt.legend()
        plt.show()
