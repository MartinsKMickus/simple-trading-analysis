



import datetime
from math import floor
import os
import random

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
        # 6.5 hours a day market is open.
        self.data_analyzer = DataAnalyzer(window=settings.DATA_WINDOW)
        self.data_analyzer.ai_manager.load_model()

    def get_predicted_change(self, symbol: str, last_date: datetime.datetime, risk_calculation: bool = False) -> tuple[datetime.date, tuple[float, float, float, float, float, float, float, float, float, float]]:
        """
        Gets last real value predicted price change in percentage and value. Based on last_real -> predicted and last_predicted -> predicted changes
        Args:
            symbol (str): Symbol name.

        Returns:
            tuple[last real, real percentage change, real value change, last predicted, predicted percentage change, predicted value change, trend precision, prediction confidence, prev value prediction confidence]
        """
        result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        data = self.data_analyzer.data_manager.get_symbol_data(symbol=symbol, interval=self.data_analyzer.interval)#, online=True)
        if max(data['time']) < last_date:
            data = self.data_analyzer.data_manager.get_symbol_data(symbol=symbol, interval=self.data_analyzer.interval, online=True)
        filtered_data = self.data_analyzer.data_manager.filter_data(data=data, date_to=last_date)
        scaled_data = self.data_analyzer.ai_manager.scale_for_ai(filtered_data)

        predictions, confidences = self.data_analyzer.ai_manager.predict_all_values(values=scaled_data[-self.data_analyzer.window-1-self.data_analyzer.ai_manager.predict_after_time:], monte_carlo=True)
        predictions = self.data_analyzer.ai_manager.scale_back_value(predictions)

        # predicted, confidence = self.data_analyzer.ai_manager.predict_next_value(values=scaled_data)
        # predicted = self.data_analyzer.ai_manager.scale_back_value(predicted)[0]

        # last_real -> predicted
        result[0] = filtered_data.iloc[-1]['close']
        result[1] = (predictions[-1]-result[0])/result[0]
        result[2] = round(predictions[-1] - result[0], 2)
        # last_predicted -> predicted. Need to get data for last real value but as if model predicts it to get rid of the model bias on specific data. As a result i-th + predict_after_time is compared to i-th value -1 because next value is actually when predict_after_time = 0
        # predicted_prev, confidence_prev = self.data_analyzer.ai_manager.predict_next_value(values=scaled_data[-self.data_analyzer.window-1-self.data_analyzer.ai_manager.predict_after_time])
        result[3] = predictions[0]
        result[4] = (predictions[-1]-result[3])/result[3]
        result[5] = round(predictions[-1] - result[3], 2)
        result[6] = self.data_analyzer.ai_manager.get_metrics_on_data(values=scaled_data, symbol=symbol)[0]
        result[7] = confidences[-1]
        result[8] = confidences[1]

        if risk_calculation:
            risk_value = 0
            values = []
            for checkup in range(1, len(predictions)):
                next_change = (predictions[checkup]-predictions[0])/predictions[0]
                values.append(next_change)
                if result[4] < 0 and next_change > risk_value:
                    risk_value = next_change
                elif result[4] > 0 and next_change < risk_value:
                    risk_value = next_change
                # if next_change > 0 and result[4] > 0 and next_change > result[4]:
                    # print(f'Possible faster target: {next_change}')
                    # break
                # if next_change < 0 and result[4] < 0 and next_change < result[4]:
                    # print(f'Possible faster target: {next_change}')
                    # break
            result[9] = risk_value
            if result[9] == 0:
                result[9] = 0.001
            plt.plot(values)
            plt.show()

        
        return (datetime.datetime.date(filtered_data.iloc[-1]["time"]), tuple(result))

    def predict_next_close(self, symbol: str, last_date = current_date):
        """
        Shows next possible price value and change for stock. Shows data information.
        Args:
            symbol (str): Symbol name.
        """
        
        result_date, result = self.get_predicted_change(symbol=symbol, last_date=last_date, risk_calculation=True)
        if result_date != last_date.date():
                print(f'{bcolors.WARNING}Last date: {result_date} of {symbol} is not correct! {last_date.date()} was chosen{bcolors.ENDC}')
        if self.data_analyzer.in_exclude(symbol=symbol):
            print(f'{bcolors.WARNING}Warning! Symbol is in except symbol list{bcolors.ENDC}')
        print(f'For stock {symbol} after: {result_date} with price: {round(result[0], 2)}')
        print(f'Last real -> Predicted change: {round(result[1]*100, 2)}% that is: {result[2]}')
        print(f'Last predicted -> Predicted change: {round(result[4]*100, 2)}% that is: {result[5]}')
        print(f'Model confidence: {round(result[7]*100, 2)}%')
        print(f'Model confidence of last predicted: {round(result[8]*100, 2)}%')
        print(f'Possible Reward/Risk ratio (predicted): {round(abs(result[5]/result[9]), 2)}')

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

    def predict_market(self, last_date: datetime.datetime = current_date, shuffle: bool = False, symbol_count: int = 0):
        """
        Gets model predictions for stock market and saves data in file.

        Prediction file will consist of columns:
        symbol;
        prediction based prediction (prediction calculated by looking at previous predicted value) percentage/price;
        real value based prediction (prediction calculated by looking at previous real value) percentage/price;
        Args:
            last_date (datetime.datetime): Date to start prediction from.
            shuffle: (bool): Go in shuffled oreder
            symbol_count: (int): Process only this number of symbols
        """
        columns = ['symbol', 'price', 'predicted_real_change', 'pred_real_percent', 'last_predicted_price', 'predicted_change', 'pred_percent']
        market_results = pd.DataFrame(columns=columns)
        if not os.path.exists(f'{settings.DATA_DIR}/predictions'):
            os.makedirs(f'{settings.DATA_DIR}/predictions')
        
        if shuffle:
            symbol_list = self.data_analyzer.data_manager.symbols.copy()
            random.shuffle(symbol_list)
        else:
            symbol_list = self.data_analyzer.data_manager.symbols
        if symbol_count <= 0:
            symbol_count = len(symbol_list)
        
        i = 1
        for symbol in tqdm.tqdm(symbol_list):
            # TODO: Implement ctrl-c stop for predict_market
            # print(f'Processing {symbol}')
            if i > symbol_count:
                break
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

            market_results.to_csv(f'{settings.DATA_DIR}/predictions/Predictions_{self.data_analyzer.interval}_{last_date.date()}_future{self.data_analyzer.ai_manager.predict_after_time + 1}.csv', index=False)
            i += 1

    def strategy_tester(self, symbol: str, graphs: bool = True, info: bool = True):
        """
        Gets model predictions for stock market and saves data in file.

        Prediction file will consist of columns:
        symbol;
        prediction based prediction (prediction calculated by looking at previous predicted value) percentage/price;
        real value based prediction (prediction calculated by looking at previous real value) percentage/price;
        Args:
            last_date (datetime.datetime): Date to start prediction from.
        Returns:
            (winrate, balance) or None
        """
        # #33
        
        if self.data_analyzer.ai_manager.diversity_threshold >= settings.MIN_REQUIRED_PROFIT * 2:
            print(f'{bcolors.FAIL}Configured lowest profit {settings.MIN_REQUIRED_PROFIT} is lower than possible result diversity on one side {self.data_analyzer.ai_manager.diversity_threshold * 2}{bcolors.ENDC}')
            return
        loaded_data = self.data_analyzer.data_manager.get_symbol_data(symbol=symbol, interval=self.data_analyzer.interval)
        scaled_data = self.data_analyzer.ai_manager.scale_for_ai(loaded_data)
        loaded_data = self.data_analyzer.data_manager.get_symbol_data(symbol=symbol, interval=self.data_analyzer.interval, do_drop=False)
        if settings.USE_MONTE_CARLO:
            predictions, confidences = self.data_analyzer.ai_manager.predict_all_values(values=scaled_data, monte_carlo=True)
        else:
            predictions = self.data_analyzer.ai_manager.predict_all_values(values=scaled_data, monte_carlo=False)
            confidences = np.ones((len(predictions)))
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
        sequence = []
        balance_history = [balance]
        win_change_histogram = []
        loss_change_histogram = []
        win_confidence_histogram = []
        loss_confidence_histogram = []
        # last_date will be date which AFTER prediction will be made. + predict_after_time because first value has to be checked agaisnt model bias. -1 because LAST REAL VALUE BEFORE PREDICTION
        last_date = loaded_data['time'].dt.date.iloc[self.data_analyzer.window + self.data_analyzer.ai_manager.predict_after_time - 1]
        # test_list_real = []
        # test_list_pred = []
        # Start with 1 because to check model bias
        for i in tqdm.tqdm(range(self.data_analyzer.ai_manager.predict_after_time + 1, len(predictions) - self.data_analyzer.ai_manager.predict_after_time), delay=5):
            # Need to check which dates are actually available for data
            if settings.FIRST_DAY_HOUR and last_date == loaded_data['time'].dt.date.iloc[self.data_analyzer.window + i]:
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
            if current_confidence < settings.MIN_CONFIDENCE or last_confidence < settings.MIN_CONFIDENCE or settings.MIN_REQUIRED_PROFIT > abs(predicted_change) or abs(predicted_change) > settings.MAX_REQUIRED_PROFIT or (settings.CASH_ACCOUNT and predicted_change < 0):
                # print(f'Today {last_date} is skipped because some of results:')
                # print(f'Confidence of last_predicted: {round(results[8]*100, 2)}%')
                # print(f'Confidence of predicted: {round(results[7]*100, 2)}%')
                # print(f'Predicted profit: {abs(round(results[4]*100, 2))}%')
                last_date = loaded_data['time'].dt.date.iloc[self.data_analyzer.window + i]
                continue
            # Dynamic predicted change based on confidence and diversity: WRONG IMPLEMENTATION
            # if predicted_change < 0:
            #     predicted_change += self.data_analyzer.ai_manager.diversity_threshold * (1 - current_confidence) * 0.5
            # else:
            #     predicted_change -= self.data_analyzer.ai_manager.diversity_threshold * (1 - current_confidence) * 0.5
            
            # Exceeding risk change means wrong prediction
            risk_change = -predicted_change/settings.REWARD_RISK_RATIO
            # If predictions predicted exceeded risk, then we will not continue
            skip = False
            for checkup in range(1, self.data_analyzer.ai_manager.predict_after_time + 1):
                next_change = (predictions[i + checkup]-predictions[i-self.data_analyzer.ai_manager.predict_after_time - 1])/predictions[i-self.data_analyzer.ai_manager.predict_after_time - 1]
                # If predictions exceeds risk (Removed and replaced with method below since model ID X4/41)
                # if risk_change < 0 and next_change < risk_change or risk_change > 0 and next_change > risk_change:
                #     last_date = loaded_data['time'].dt.date.iloc[self.data_analyzer.window + i]
                #     # print(f"With confidence {round(current_confidence, 2)}% risk was exceeded")
                #     skip = True
                #     break
                # If predictions goes to other side (loss)
                if risk_change < 0 and next_change < 0 or risk_change > 0 and next_change > 0:
                    last_date = loaded_data['time'].dt.date.iloc[self.data_analyzer.window + i]
                    # print(f"With confidence {round(current_confidence, 2)}% risk was exceeded")
                    skip = True
                    break
                if next_change > 0 and predicted_change > 0 and next_change > predicted_change:
                    break
                if next_change < 0 and predicted_change < 0 and next_change < predicted_change:
                    break
            if skip:
                continue

            change = 0
            no_decision = True
            
            # real_data[window -> i-th] predicts real_data[i-th + predict_after_time]. + 1 because range stop is not included
            # Example: At i=0 prediction is for real_data[i:window], window is NOT INCLUDED so checkup starts at window + predict_after_time
            future_exceeded = 0
            wait_to_buy = settings.WAIT_FOR_ENTRY
            entry_price = loaded_data.iloc[self.data_analyzer.window + i - 1]['close']
            entry_price += entry_price * risk_change * settings.BETTER_ENTRY
            # WRONG ASSUMPTION -> Need to start not from next candle but from one after because of possible price fluctuation <- WRONG ASSUMPTION
            for real_data in range(self.data_analyzer.window + i, len(predictions) + self.data_analyzer.window): # self.data_analyzer.window + i + self.data_analyzer.ai_manager.predict_after_time
                # test_list_pred.append(predictions[i])
                # test_list_real.append(loaded_data.iloc[real_data]['close'])
                # Last value fed into model was window. - 1 because window value is already predicted
                real_max = loaded_data.iloc[real_data]['high']
                real_min = loaded_data.iloc[real_data]['low']
                real_max_change = (real_max-entry_price)/entry_price
                real_min_change = (real_min-entry_price)/entry_price
                
                # Need to get stock value reach entry price before enterint
                if wait_to_buy > 0:
                    if entry_price < real_max and entry_price > real_min:
                        # Entry
                        wait_to_buy = -1
                    else:
                        # Wait next time unit to enter
                        wait_to_buy -= 1
                        continue
                elif wait_to_buy == 0:
                    # No entry price was reached within period
                    last_date = loaded_data['time'].dt.date.iloc[self.data_analyzer.window + i]
                    break
                
                if risk_change < 0 and real_min_change <= risk_change:
                    # print(f'Guess for {last_date} is wrong:')
                    # print(f'Confidence: {round(current_confidence*100, 2)}%')
                    # print(f'Predicted change: {round(predicted_change*100, 2)}% but risk {round(risk_change*100, 2)}% was exceeded {round(real_min_change*100, 2)}%')
                    change = risk_change
                    no_decision = False
                    break
                elif risk_change > 0 and real_max_change >= risk_change:
                    # print(f'Guess for {last_date} is wrong:')
                    # print(f'Confidence: {round(current_confidence*100, 2)}%')
                    # print(f'Predicted change: {round(predicted_change*100, 2)}% but risk {round(risk_change*100, 2)}% was exceeded {round(real_max_change*100, 2)}%')
                    change = -risk_change
                    no_decision = False
                    break

                if predicted_change < 0 and real_min_change <= predicted_change:
                    change = -predicted_change
                    no_decision = False
                    break
                elif predicted_change > 0 and real_max_change >= predicted_change:
                    change = predicted_change
                    no_decision = False
                    break
                future_exceeded += 1
                # if future_exceeded > self.data_analyzer.ai_manager.predict_after_time + 1 + settings.TIME_UNITS_WAIT:
                #     # print(f'Prediction with confidence {round(current_confidence*100, 2)}% Pred. ch: {round(predicted_change*100, 2)}% exceeded TU: {future_exceeded}')
                #     break

            if wait_to_buy != -1:
                continue

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
            # stock_val = loaded_data.iloc[self.data_analyzer.window + i - 1]['close']
            # print(f'At: {last_date}')
            # print(f'Stock val: {stock_val}')
            risk = round(settings.ACCOUNT_RISK * balance / 5) * 5
            qty_invested = int(risk / (entry_price * abs(risk_change)))
            # print(f'QTY: {qty_invested}')
            # print(f'Predicted: {predicted_change}')
            # print(f'Change: {change}')
            value_invested = entry_price * qty_invested
            if value_invested > balance:
                # Risked value shouldn't be exceeded
                if change > 0:
                    correct_guesses -= 1
                else:
                    wrong_guesses -= 1
                last_date = loaded_data['time'].dt.date.iloc[self.data_analyzer.window + i]
                continue
                qty_invested_new = floor(balance / entry_price)
                # print(f'Adjusted quantity invested {qty_invested} -> {qty_invested_new} because it exceeded balance {round(balance, 2)} < {round(value_invested, 2)}')
                # print(f'Balance used now: {stock_val * qty_invested_new}')
                qty_invested = qty_invested_new
            balance += entry_price * qty_invested * change
            balance_history.append(balance)
            if change < 0:
                loss_confidence_histogram.append(current_confidence)
                loss_change_histogram.append(predicted_change)
            else:
                win_confidence_histogram.append(current_confidence)
                win_change_histogram.append(predicted_change)
            last_date = loaded_data['time'].dt.date.iloc[self.data_analyzer.window + i]
        
        if graphs:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            fig.subplots_adjust(hspace=0.4)
            # Determine the common data range for both sets
            try:
                confidence_range = (min(min(win_confidence_histogram), min(loss_confidence_histogram)), max(max(win_confidence_histogram), max(loss_confidence_histogram)))
            except:
                confidence_range = None
            try:
                change_range = (min(min(win_change_histogram), min(loss_change_histogram)), max(max(win_change_histogram), max(loss_change_histogram)))
            except:
                change_range = None
            ax1.hist(win_confidence_histogram, bins=50, range=confidence_range, label='Changes that won', density=True, alpha=0.7)
            ax1.hist(loss_confidence_histogram, bins=50, range=confidence_range, label='Changes that lost', density=True, alpha=0.7)
            ax2.hist(win_change_histogram, bins=50, range=change_range, label='Changes that won', density=True, alpha=0.7)
            ax2.hist(loss_change_histogram, bins=50, range=change_range, label='Changes that lost', density=True, alpha=0.7)
            ax3.plot(balance_history)
            ax1.title.set_text('Confidence')
            ax2.title.set_text('Predicted change')
            ax3.title.set_text('Balance History')
            # plt.plot(test_list_pred, label='Predicted')
            # plt.plot(test_list_real, label='Actual')
            plt.legend()
            plt.show()
        
        if info:
            print(f'For symbol {symbol}:')
        if correct_guesses+wrong_guesses > 0:
            winrate = correct_guesses/(correct_guesses + wrong_guesses)
            if info:
                print(f'Strategy was correct with accuracy {round(winrate*100,2)}%')
        else:
            if info:
                print('No trades would be executed')
            return ('-', settings.START_BALANCE)
        if info:
            print(f'Correct guess count: {correct_guesses}')
            print(f'Wrong guess count: {wrong_guesses}')
            print(f'Using strategy balance change: {settings.START_BALANCE} -> ', end='')
            if balance < settings.START_BALANCE:
                print(bcolors.FAIL, end='')
            else:
                print(bcolors.OKGREEN, end='')
            print(f'{round(balance, 2)}{bcolors.ENDC}')
        return (winrate, balance)

    def model_performance_summary(self):
        """
        Gets model performance summary for specified stocks in settings file
        """
        results = {}
        for symbol in tqdm.tqdm(settings.TEST_STOCKS):
            # print(f'Analyzing {symbol}')
            results[symbol] = self.strategy_tester(symbol=symbol, graphs=False, info=False)

        print('Results:')
        for key in results.keys():
            print(key, end=' ')
        print()
        for result in results.values():
            winrate = result[0]
            if not isinstance(winrate, str):
                winrate = round(winrate*100, 2)
            print(f'{winrate}%,{round(result[1], 2)}€', end=',')
        print()