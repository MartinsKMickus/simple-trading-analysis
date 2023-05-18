

import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tradinga import settings

from tradinga.ai_helper import get_evaluation, get_xy_arrays, model_v3, predict_simple_next_values, scale_for_ai, test_simple_model, train_model
from tradinga.ai_models import MODEL_METRICS, load_model
from tradinga.data_helper import download_newest_data, load_existing_data, save_data_to_csv


def make_model(data: pd.DataFrame, look_back: int, epochs, model_path: str, test_data: pd.DataFrame = None, log_symbol_name=None):
    data.sort_values('time', inplace=True)
    data['time'] = pd.to_datetime(data['time'])
    scaled_data = scale_for_ai(data=data['close'])

    # prepare feature and labels
    x_train, y_train = get_xy_arrays(values=scaled_data, window=look_back)

    if isinstance(test_data, pd.DataFrame):
        test_data.sort_values('time', inplace=True)
        test_data['time'] = pd.to_datetime(data['time'])

        test_scaled = scale_for_ai(data=test_data['close'])

        x_test, y_test = get_xy_arrays(values=test_scaled, window=look_back)

    if os.path.isdir(model_path):
        print("Found such model. Will continue to train it.")
        model = load_model(model_path)
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
    model = load_model(model_path)
    output_data = test_simple_model(data=data,model=model,look_back=input_window)
    plot_data = data[['time', 'close']].copy()

    plot_data = plot_data.iloc[input_window:]
    plot_data['predicted'] = np.concatenate(output_data)#[:plot_data.shape[0]]

    plt.plot(plot_data['time'], plot_data['close'], label='Actual')
    plt.plot(plot_data['time'], plot_data['predicted'], label='Predicted')
    plt.legend()
    plt.show()


def draw_future(model_path: str, input_window:int, data: pd.DataFrame, predict: int):
    if not os.path.isdir(model_path):
        print(f"Model {model_path} doesn't exist")
        return
    
    model = load_model(model_path)
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

    scaled_data = scale_for_ai(data=data['close'])

    x_test, y_test = get_xy_arrays(values=scaled_data, window=input_window)

    return get_evaluation(x_test=x_test, y_test=y_test,model=model)


def analyze_model_metrics(model_path: str, input_window:int, online: bool = False):
    """
    Uses symbols to get model metrics
    """
    if not os.path.isdir(model_path):
        print(f"Model {model_path} doesn't exist")
        return
    model = load_model(model_path)

    symbols = load_existing_data(None, settings.INTERVAL)
    if not isinstance(symbols, pd.DataFrame):
        if not online:
            print("Empty symbols file. Run in online mode to download it.")
            return
        from tradinga.api_helper import get_nasdaq_symbols
        print("Empty symbols file. Trying to download it now.")
        symbols = get_nasdaq_symbols()
        save_data_to_csv(data=symbols)

    metric_values = [[] for _ in range(len(MODEL_METRICS))]

    total_symbols = symbols['symbol'].size
    current_pos = 0
    for symbol in symbols['symbol']:
        current_pos += 1
        if online:
            download_newest_data(symbol=symbol,interval=settings.INTERVAL)
        data = load_existing_data(symbol=symbol, interval=settings.INTERVAL)
        if not isinstance(data, pd.DataFrame):
            continue
        if len(data) < input_window:
            print(f"{symbol} has not enough data points")
            continue

        metrics = get_model_metrics(data=data, model=model, input_window=input_window)
        if not isinstance(metrics, list):
            raise Exception("Looks like there is only one model metric or something is wrong!")
        if len(metrics) != len(MODEL_METRICS):
            raise Exception("Different count of model metrics detected!")
        
        for metric in range(len(MODEL_METRICS)):
            metric_values[metric].append(metrics[metric])
        print(f'Finished symbol: {symbol}')
        print(f'Progress: {round(current_pos/total_symbols*100, 2)}%')
    print(f'Progress: {round(current_pos/total_symbols*100, 2)}%')

    print(f'For model: ({model_path}):')
    for metric in range(len(MODEL_METRICS)):
        print(f'{MODEL_METRICS[metric]} AVG: {sum(metric_values[metric])/len(metric_values[metric])}')


def analyze_market_stocks(model_path: str, input_window:int, future: int = 10, download_all: bool = False):
    if not os.path.isdir(model_path):
        print(f"Model {model_path} doesn't exist")
        return
    model = load_model(model_path)

    symbols = load_existing_data(None, settings.INTERVAL)
    if not isinstance(symbols, pd.DataFrame):
        # if not online:
        #     print("Empty symbols file. Run in online mode to download it.")
        #     return
        from tradinga.api_helper import get_nasdaq_symbols
        print("Empty symbols file. Trying to download it now.")
        symbols = get_nasdaq_symbols()
        save_data_to_csv(data=symbols)

    max_growth = 0
    stonks_symbol = ''
    total_symbols = symbols['symbol'].size
    current_pos = 0
    for symbol in symbols['symbol']:
        current_pos += 1
        if download_all:
            try:
                download_newest_data(symbol=symbol,interval=settings.INTERVAL)
            except:
                print(f'Failed to download {symbol}. Skipping...')
                continue
        
        data = load_existing_data(symbol=symbol, interval=settings.INTERVAL)
        if not isinstance(data, pd.DataFrame):
            continue

        try:
            download_newest_data(symbol=symbol,interval=settings.INTERVAL)
        except:
            print(f'Failed to download {symbol}. Skipping...')
            continue
        data = load_existing_data(symbol=symbol, interval=settings.INTERVAL)
        if len(data) < input_window:
            print(f"{symbol} has not enough data points")
            continue

        metrics = get_model_metrics(data=data, model=model, input_window=input_window)
        if not isinstance(metrics, list):
            raise Exception("Looks like there is only one model metric or something is wrong!")
        # if len(metrics) != len(MODEL_METRICS):
        #     raise Exception("Different count of model metrics detected!")
        
        # First metric will be loss value
        if metrics[0] < 0.01:
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
