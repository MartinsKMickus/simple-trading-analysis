



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
        # 7 hours a day market is open. 100 days around 700 hours -> 600
        self.data_analyzer = DataAnalyzer(window=600)
        self.data_analyzer.ai_manager.load_model()

    def get_predicted_change(self, symbol: str, last_date: datetime.datetime) -> tuple[datetime.date, tuple[float, float, float, float, float, float, float, float, float]]:
        """
        Gets last real value predicted price change in percentage and value. Based on last_real -> predicted and last_predicted -> predicted changes
        Args:
            symbol (str): Symbol name.

        Returns:
            tuple[last real, real percentage change, real value change, last predicted, predicted percentage change, predicted value change, trend precision, prediction confidence, prev value prediction confidence]
        """
        result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        data = self.data_analyzer.data_manager.get_symbol_data(symbol=symbol, interval=self.data_analyzer.interval)#, online=True)
        if max(data['time']) < last_date:
            data = self.data_analyzer.data_manager.get_symbol_data(symbol=symbol, interval=self.data_analyzer.interval, online=True)
        filtered_data = self.data_analyzer.data_manager.filter_data(data=data, date_to=last_date)
        scaled_data = self.data_analyzer.ai_manager.scale_for_ai(filtered_data)

        predicted, confidence = self.data_analyzer.ai_manager.predict_next_value(values=scaled_data)
        predicted = self.data_analyzer.ai_manager.scale_back_value(predicted)[0]

        # last_real -> predicted
        result[0] = filtered_data.iloc[-1]['close']
        result[1] = (predicted-result[0])/result[0]
        result[2] = round(predicted - result[0], 2)
        # last_predicted -> predicted. Need to get data for last real value but as if model predicts it to get rid of the model bias on specific data. As a result i-th + predict_after_time is compared to i-th value -1 because next value is actually when predict_after_time = 0
        predicted_prev, confidence_prev = self.data_analyzer.ai_manager.predict_next_value(values=scaled_data[:-1-self.data_analyzer.ai_manager.predict_after_time])
        result[3] = self.data_analyzer.ai_manager.scale_back_value(predicted_prev)[0]
        result[4] = (predicted-result[3])/result[3]
        result[5] = round(predicted - result[3], 2)
        result[6] = self.data_analyzer.ai_manager.get_metrics_on_data(values=scaled_data, symbol=symbol)[0]
        result[7] = confidence
        result[8] = confidence_prev

        
        return (datetime.datetime.date(filtered_data.iloc[-1]["time"]), tuple(result))

    def predict_next_close(self, symbol: str, last_date = current_date):
        """
        Shows next possible price value and change for stock. Shows data information.
        Args:
            symbol (str): Symbol name.
        """
        
        result_date, result = self.get_predicted_change(symbol=symbol, last_date=last_date)
        if result_date != last_date.date():
                print(f'{bcolors.WARNING}Last date: {result_date} of {symbol} is not correct! {last_date.date()} was chosen{bcolors.ENDC}')
        if self.data_analyzer.in_exclude(symbol=symbol):
            print(f'{bcolors.WARNING}Warning! Symbol is in except symbol list{bcolors.ENDC}')
        print(f'For stock {symbol} after: {result_date} with price: {round(result[0], 2)}')
        print(f'Last real -> Predicted change: {round(result[1]*100, 2)}% that is: {result[2]}')
        print(f'Last predicted -> Predicted change: {round(result[4]*100, 2)}% that is: {result[5]}')
        print(f'Model confidence: {round(result[7]*100, 2)}%')
        print(f'Model confidence of last predicted: {round(result[8]*100, 2)}%')

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

    def predict_market(self, last_date: datetime.datetime = current_date):
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
        if not os.path.exists(f'{settings.DATA_DIR}/predictions'):
            os.makedirs(f'{settings.DATA_DIR}/predictions')
        # i = 0
        for symbol in tqdm.tqdm(self.data_analyzer.data_manager.symbols):
            # TODO: Implement ctrl-c stop for predict_market
            # print(f'Processing {symbol}')
            # if i > 20:
            #     break
            # i += 1
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
                'trend_precision': [results[6]],
                'prediction confidence': [results[7]],
                'prev value prediction confidence': [results[8]]
            })
            # print(new_row)
            market_results = pd.concat([market_results, new_row], ignore_index=True)
            # print(market_results)

        
            market_results.to_csv(f'{settings.DATA_DIR}/predictions/Predictions_{self.data_analyzer.interval}_{last_date.date()}_future{self.data_analyzer.ai_manager.predict_after_time}.csv', index=False)

    def strategy_tester(self, symbol: str):
        """
        Gets model predictions for stock market and saves data in file.

        Prediction file will consist of columns:
        symbol;
        prediction based prediction (prediction calculated by looking at previous predicted value) percentage/price;
        real value based prediction (prediction calculated by looking at previous real value) percentage/price;
        Args:
            last_date (datetime.datetime): Date to start prediction from.
        """
        # #33
        cash_account = True
        if self.data_analyzer.ai_manager.diversity_threshold >= settings.MIN_REQUIRED_PROFIT:
            print(f'{bcolors.FAIL}Configured lowest profit {settings.MIN_REQUIRED_PROFIT} is lower than possible result diversity {self.data_analyzer.ai_manager.diversity_threshold}{bcolors.ENDC}')
            return
        loaded_data = self.data_analyzer.data_manager.get_symbol_data(symbol=symbol, interval=self.data_analyzer.interval)
        scaled_data = self.data_analyzer.ai_manager.scale_for_ai(loaded_data)
        predictions, confidences = self.data_analyzer.ai_manager.predict_all_values(values=scaled_data, monte_carlo=True)
        predictions = self.data_analyzer.ai_manager.scale_back_value(predictions)
        # plot_data = loaded_data.iloc[self.data_analyzer.window:]
        # plot_data['predicted'] = predictions
        # # loaded_data = loaded_data.iloc[self.data_analyzer.window:]
        # print(len(predictions))
        # print(len(loaded_data))
        # plt.plot(plot_data['time'], plot_data['close'], label='Actual')
        # plt.plot(plot_data['time'], plot_data['predicted'], label='Predicted')
        # plt.legend()
        # plt.show()
        balance = settings.START_BALANCE
        correct_guesses = 0
        wrong_guesses = 0
        # last_date will be date which AFTER prediction will be made. + predict_after_time because first value has to be checked agaisnt model bias. -1 because LAST REAL VALUE BEFORE PREDICTION
        last_date = loaded_data['time'].dt.date.iloc[self.data_analyzer.window + self.data_analyzer.ai_manager.predict_after_time - 1]
        # test_list_real = []
        # test_list_pred = []
        # Start with 1 because to check model bias
        for i in tqdm.tqdm(range(self.data_analyzer.ai_manager.predict_after_time + 1, len(predictions) - self.data_analyzer.ai_manager.predict_after_time)):
            # Need to check which dates are actually available for data
            if last_date == loaded_data['time'].dt.date.iloc[self.data_analyzer.window + i]:
                continue
            # result_date, results = self.get_predicted_change(symbol=symbol, last_date=datetime.datetime.combine(last_date, datetime.datetime.min.time()))
            # predicted change is change from predictions[i-th] -> predictions[i-th + 1 + predict_after_time] where predict_after_time = 0 if next value is predicted
            predicted_change = (predictions[i]-predictions[i-self.data_analyzer.ai_manager.predict_after_time - 1])/predictions[i-self.data_analyzer.ai_manager.predict_after_time - 1]
            current_confidence = confidences[i]
            last_confidence = confidences[i-self.data_analyzer.ai_manager.predict_after_time - 1]
            
            # if result_date != last_date:
            #     print(f'{bcolors.FAIL}ERROR! Data incosistency deteted for {symbol}!')
            #     print(f'Asked for date: {last_date} but got {result_date}{bcolors.ENDC}')
            #     return
            if current_confidence < settings.MIN_CONFIDENCE or last_confidence < settings.MIN_CONFIDENCE or settings.MIN_REQUIRED_PROFIT > abs(predicted_change) or abs(predicted_change) > settings.MAX_REQUIRED_PROFIT or (cash_account and predicted_change < 0):
                # print(f'Today {last_date} is skipped because some of results:')
                # print(f'Confidence of last_predicted: {round(results[8]*100, 2)}%')
                # print(f'Confidence of predicted: {round(results[7]*100, 2)}%')
                # print(f'Predicted profit: {abs(round(results[4]*100, 2))}%')
                last_date = loaded_data['time'].dt.date.iloc[self.data_analyzer.window + i]
                continue
            # Dynamic predicted change based on confidence and diversity:
            if predicted_change < 0:
                predicted_change += self.data_analyzer.ai_manager.diversity_threshold * (1 - current_confidence) * 0.5
            else:
                predicted_change -= self.data_analyzer.ai_manager.diversity_threshold * (1 - current_confidence) * 0.5
            
            # Exceeding risk change means wrong prediction
            risk_change = -predicted_change/settings.REWARD_RISK_RATIO
            change = 0
            no_decision = True
            
            # real_data[window -> i-th] predicts real_data[i-th + predict_after_time]. + 1 because range stop is not included
            # Example: At i=0 prediction is for real_data[i:window], window is NOT INCLUDED so checkup starts at window + predict_after_time
            for real_data in range(self.data_analyzer.window + i, self.data_analyzer.window + i + self.data_analyzer.ai_manager.predict_after_time):
                # test_list_pred.append(predictions[i])
                # test_list_real.append(loaded_data.iloc[real_data]['close'])
                # Last value fed into model was window. - 1 because window value is already predicted
                real_max_change = (loaded_data.iloc[real_data]['high']-loaded_data.iloc[self.data_analyzer.window + i - 1]['close'])/loaded_data.iloc[self.data_analyzer.window + i - 1]['close']
                real_min_change = (loaded_data.iloc[real_data]['low']-loaded_data.iloc[self.data_analyzer.window + i - 1]['close'])/loaded_data.iloc[self.data_analyzer.window + i - 1]['close']

                if risk_change < 0 and real_min_change <= risk_change:
                    # print(f'Guess for {last_date} is wrong:')
                    # print(f'Confidence: {round(current_confidence*100, 2)}%')
                    # print(f'Predicted change: {round(predicted_change*100, 2)}% but risk {round(risk_change*100, 2)}% was exceeded {round(real_min_change*100, 2)}%')
                    wrong_guesses += 1
                    change = risk_change
                    no_decision = False
                    break
                elif risk_change > 0 and real_max_change >= risk_change:
                    # print(f'Guess for {last_date} is wrong:')
                    # print(f'Confidence: {round(current_confidence*100, 2)}%')
                    # print(f'Predicted change: {round(predicted_change*100, 2)}% but risk {round(risk_change*100, 2)}% was exceeded {round(real_max_change*100, 2)}%')
                    wrong_guesses += 1
                    change = -risk_change
                    no_decision = False
                    break

                if predicted_change < 0 and real_min_change <= predicted_change:
                    correct_guesses += 1
                    change = -predicted_change
                    no_decision = False
                    break
                elif predicted_change > 0 and real_max_change >= predicted_change:
                    correct_guesses += 1
                    change = predicted_change
                    no_decision = False
                    break
            if no_decision:
                # If no extreme was reached then last value is considered profit or loss. - 1 because window value is already predicted
                change = (loaded_data.iloc[self.data_analyzer.window + i + self.data_analyzer.ai_manager.predict_after_time]['close']-loaded_data.iloc[self.data_analyzer.window + i - 1]['close'])/loaded_data.iloc[self.data_analyzer.window + i - 1]['close']
                if risk_change > 0:
                    change = -change
                if change < 0:
                    wrong_guesses += 1
                    # print(f'Guess for {last_date} is wrong:')
                else:
                    correct_guesses += 1
            balance += balance * change
            last_date = loaded_data['time'].dt.date.iloc[self.data_analyzer.window + i]

        # plt.plot(test_list_pred, label='Predicted')
        # plt.plot(test_list_real, label='Actual')
        # plt.legend()
        # plt.show()
        print(f'For symbol {symbol}:')
        if correct_guesses+wrong_guesses > 0:
            print(f'Strategy was correct with accuracy {round(correct_guesses/(correct_guesses + wrong_guesses)*100,2)}%')
        else:
            print('No trades would be executed')
        print(f'Correct guess count: {correct_guesses}')
        print(f'Wrong guess count: {wrong_guesses}')
        print(f'Using strategy balance change: {settings.START_BALANCE} -> ', end='')
        if balance < settings.START_BALANCE:
            print(bcolors.FAIL, end='')
        else:
            print(bcolors.OKGREEN, end='')
        print(f'{round(balance, 2)}{bcolors.ENDC}')
