

import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tradinga import constants

from tradinga.ai_helper import get_evaluation, model_v3, predict_simple_next_values, test_simple_model, train_model
from tradinga.ai_models import mape_loss
from tradinga.data_helper import download_newest_data, load_existing_data, save_data_to_csv


def make_model(data: pd.DataFrame, look_back: int, epochs, model_path: str, test_data: pd.DataFrame = None, log_symbol_name=None):
    data.sort_values('time', inplace=True)
    data['time'] = pd.to_datetime(data['time'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

    # prepare feature and labels
    x_train = []
    y_train = []

    for i in range(look_back, len(scaled_data)):
        x_train.append(scaled_data[i-look_back:i, 0])
        y_train.append(scaled_data[i, 0])  # To predict next value for training

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    if isinstance(test_data, pd.DataFrame):
        test_data.sort_values('time', inplace=True)
        test_data['time'] = pd.to_datetime(data['time'])

        test_scaled = scaler.fit_transform(test_data['close'].values.reshape(-1, 1))

        # prepare feature and labels
        x_test = []
        y_test = []

        for i in range(look_back, len(test_scaled)):
            x_test.append(test_scaled[i-look_back:i, 0])
            y_test.append(test_scaled[i, 0])  # To predict next value for training

        x_test, y_test = np.array(x_test), np.array(y_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    if os.path.isdir(model_path):
        print("Found such model. Will continue to train it.")
        #sleep(5)
        with tf.keras.utils.custom_object_scope({'mape_loss': mape_loss}):
            model = tf.keras.models.load_model(model_path)
    else:
        model = model_v3(x_train.shape[1])
    if isinstance(test_data, pd.DataFrame):
        train_model(model=model,x_train=x_train,y_train=y_train,epochs=epochs, x_test=x_test, y_test=y_test, log_name=log_symbol_name)
    else:
        train_model(model=model,x_train=x_train,y_train=y_train,epochs=epochs, log_name=log_symbol_name)
    model.save(model_path)
    

def test_model_performance(model_path: str, input_window:int, data: pd.DataFrame, start_ammount: float):
    if not os.path.isdir(model_path):
        print(f"Model {model_path} doesn't exist")
        return
    with tf.keras.utils.custom_object_scope({'mape_loss': mape_loss}):
        model = tf.keras.models.load_model(model_path)
    output_data = test_simple_model(data=data,model=model,look_back=input_window)
    #positive_feedback = 0
    #buy = False
    #predicted_values = []
    #money_change = []
    #money_change.append(start_ammount)
    # for x in range(input_window + 1, len(data), 10): # + 1 because of profit calculations
    #     output_data = predict_simple_next_values(data=data.iloc[x-input_window:x], model=model,look_back=input_window,next=10)
    #     predicted_values.append(output_data)
    #     real_change = data['close'].iloc[x] / data['close'].iloc[x - 1]
    #     predicted_change = output_data[-1] / data['close'].iloc[x - 1]
    #print(f'Positive: {positive_feedback/len(output_data)}')
    #print(f'Money: {start_ammount}')
    plot_data = data[['time', 'close']].copy()
    #print(np.concatenate(predicted_values))
    #money_change.reverse()
    plot_data = plot_data.iloc[input_window:]
    plot_data['predicted'] = np.concatenate(output_data)#[:plot_data.shape[0]]
    #plot_data['money_change'] = money_change

    # mse, mae = get_model_metrics(data=data, model=model, input_window=input_window)
    # print(f'WARNING!\n DETECTED MSE: {mse}\n DETECTED MAE: {mae}')

    plt.plot(plot_data['time'], plot_data['close'], label='Actual')
    plt.plot(plot_data['time'], plot_data['predicted'], label='Predicted')
    #plt.plot(plot_data['time'], plot_data['money_change'], label='Profit?')
    plt.legend()
    plt.show()


def draw_future(model_path: str, input_window:int, data: pd.DataFrame, predict: int):
    if not os.path.isdir(model_path):
        print(f"Model {model_path} doesn't exist")
        return
    with tf.keras.utils.custom_object_scope({'mape_loss': mape_loss}):
        model = tf.keras.models.load_model(model_path)
    data.sort_values('time', inplace=True)
    output_data = predict_simple_next_values(data=data.iloc[len(data)-input_window:len(data)], model=model,look_back=input_window,next=predict)

    output_data = np.squeeze(output_data) # We got rows as data in column not one row
    predicted = pd.DataFrame({'time': pd.date_range(start=data['time'].max(), periods=predict+1, freq='30min')[1:], 'close': output_data})

    # mse, mae = get_model_metrics(data=data, model=model, input_window=input_window)
    # print(f'WARNING!\n DETECTED MSE: {mse}\n DETECTED MAE: {mae}')

    plt.plot(data['time'], data['close'], label='Actual')
    plt.plot(predicted['time'], predicted['close'], label='Predicted')
    plt.legend()
    plt.show()


def get_model_metrics(data: pd.DataFrame, model: tf.keras.models.Sequential, input_window: int):
    if not isinstance(data, pd.DataFrame):
        print(f"DataFrame was none")
        return
    if len(data) < input_window:
        print(f"DataFrame has not enough data points")
        return

    data.sort_values('time', inplace=True)
    data['time'] = pd.to_datetime(data['time'])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data['close'].values.reshape(-1, 1))

    # prepare feature and labels
    x_test = []
    y_test = []

    for i in range(input_window, len(data)):
        x_test.append(scaled[i-input_window:i, 0])
        y_test.append(scaled[i, 0])  # To predict next value for training

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return get_evaluation(x_test=x_test, y_test=y_test,model=model)


def analyze_model_metrics(model_path: str, input_window:int, online: bool = False):
    """
    Uses symbols to get model metrics
    """
    if not os.path.isdir(model_path):
        print(f"Model {model_path} doesn't exist")
        return
    model = tf.keras.models.load_model(model_path)

    symbols = load_existing_data(None, constants.INTERVAL)
    if not isinstance(symbols, pd.DataFrame):
        if not online:
            print("Empty symbols file. Run in online mode to download it.")
            return
        from tradinga.api_helper import alpha_vantage_list
        print("Empty symbols file. Trying to download it now.")
        symbols = alpha_vantage_list()
        save_data_to_csv(symbol=None, data=symbols, interval=None)

    mse_max = 0
    mse_min = None
    mae_max = 0
    mae_min = None
    mse_values = []
    mae_values = []

    total_symbols = symbols['symbol'].size
    current_pos = 0
    for symbol in symbols['symbol']:
        current_pos += 1
        if online:
            download_newest_data(symbol=symbol,interval=constants.INTERVAL)
        data = load_existing_data(symbol=symbol, interval=constants.INTERVAL)
        if not isinstance(data, pd.DataFrame):
            continue
        if len(data) < input_window:
            print(f"{symbol} has not enough data points")
            continue

        mse, mae = get_model_metrics(data=data, model=model, input_window=input_window)

        mse_values.append(mse)
        mae_values.append(mae)
        if mse_max < mse:
            mse_max = mse
        if mse_min == None or mse_min > mse:
            mse_min = mse
        
        if mae_max < mae:
            mae_max = mae
        if mae_min == None or mae_min > mae:
            mae_min = mae
        print(f'Finished symbol: {symbol}')
        print(f'Progress: {round(current_pos/total_symbols*100, 2)}%')
    print(f'Progress: {round(current_pos/total_symbols*100, 2)}%')

    mse_average = sum(mse_values) / len(mse_values)
    mae_average = sum(mae_values) / len(mae_values)
    print(f'For model: ({model_path}):')
    print(f'MSE min: {round(mse_min, 5)}')
    print(f'MSE max: {round(mse_max, 5)}')
    print(f'MSE avg: {round(mse_average, 5)}')
    print(f'MAE min: {round(mae_min, 5)}')
    print(f'MAE max: {round(mae_max, 5)}')
    print(f'MAE avg: {round(mae_average, 5)}')


def analyze_market_stocks(model_path: str, input_window:int, future: int = 10, download_all: bool = False):
    if not os.path.isdir(model_path):
        print(f"Model {model_path} doesn't exist")
        return
    model = tf.keras.models.load_model(model_path)

    symbols = load_existing_data(None, constants.INTERVAL)
    if not isinstance(symbols, pd.DataFrame):
        # if not online:
        #     print("Empty symbols file. Run in online mode to download it.")
        #     return
        from tradinga.api_helper import alpha_vantage_list
        print("Empty symbols file. Trying to download it now.")
        symbols = alpha_vantage_list()
        save_data_to_csv(symbol=None, data=symbols, interval=None)

    max_growth = 0
    stonks_symbol = ''
    total_symbols = symbols['symbol'].size
    current_pos = 0
    for symbol in symbols['symbol']:
        current_pos += 1
        if download_all:
            try:
                download_newest_data(symbol=symbol,interval=constants.INTERVAL)
            except:
                print(f'Failed to download {symbol}. Skipping...')
                continue
        
        data = load_existing_data(symbol=symbol, interval=constants.INTERVAL)
        if not isinstance(data, pd.DataFrame):
            continue

        try:
            download_newest_data(symbol=symbol,interval=constants.INTERVAL)
        except:
            print(f'Failed to download {symbol}. Skipping...')
            continue
        data = load_existing_data(symbol=symbol, interval=constants.INTERVAL)
        if len(data) < input_window:
            print(f"{symbol} has not enough data points")
            continue

        mse, mae = get_model_metrics(data=data, model=model, input_window=input_window)
        if mae < 0.01:
            output_data = predict_simple_next_values(data=data.iloc[len(data)-input_window:len(data)], model=model,look_back=input_window,next=future)
            output_data = np.squeeze(output_data) # We got rows as data in column not one row

            growth = output_data[-1]/data['close'].iloc[-1] # predicted / last price
            if max_growth < growth:
                max_growth = growth
                stonks_symbol = symbol[:]
        
        print(f'Finished symbol: {symbol}')
        print(f'Progress: {round(current_pos/total_symbols*100, 2)}%')
    print(f'Progress: {round(current_pos/total_symbols*100, 2)}%')

    print(f'On symbol {stonks_symbol} there is possible growth {round(max_growth*100-100, 2)}% after {future} time units')
