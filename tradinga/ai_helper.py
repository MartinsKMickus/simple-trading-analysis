import datetime
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import tradinga.constants as constants
from tradinga.utils_helper import query_yes_no

SIMPLE_MODEL_NAME = 'simple_model'
ADVANCED_MODEL_NAME = 'advanced_model'


def model_v1(i_shape, output = 1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=64,
                                    return_sequences=True,
                                    input_shape=(i_shape, 1)))
    model.add(tf.keras.layers.LSTM(units=64))
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(output))
    return model
    
def model_v2(i_shape, output = 1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=128, input_shape=(i_shape, 1)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=output))
    return model


def train_simple_model(data: pd.DataFrame, look_back: int = 100, epochs: int = 50):
    """
    Train simple model on some data

    Args:
        data (pandas.DataFrame)
        n_steps (int): The number of previous data points to use for prediction.

    Returns:
        model
    Saves data in data directory
    """
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

    if os.path.isdir(f'{constants.AI_DIR}/{SIMPLE_MODEL_NAME}_{look_back}'):
        model = tf.keras.models.load_model(
            f'{constants.AI_DIR}/{SIMPLE_MODEL_NAME}_{look_back}') # f'{constants.AI_DIR}/{SIMPLE_MODEL_NAME}_{x_train.shape[1]}'
    else:
        model = model_v2(x_train.shape[1])
        model.summary()
        model.compile(optimizer='adam',
                    loss='mean_squared_error')

        # TODO: Implement this:
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        model.fit(x_train,
                  y_train,
                  epochs=epochs)
        model.save(
            f'{constants.AI_DIR}/{SIMPLE_MODEL_NAME}_{look_back}')
        
    return model


def train_advanced_model(data: pd.DataFrame, look_back: int = 100, predict: int = 10, epochs: int = 50):
    """
    Train advanced model on some data

    Args:
        data (pandas.DataFrame)
        look_back: How much values to look back
        predict: How much values to predict in future
        n_steps (int): The number of previous data points to use for prediction.

    Returns:
        model
    Saves data in data directory
    """
    data.sort_values('time', inplace=True)
    data['time'] = pd.to_datetime(data['time'])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

    # prepare feature and labels
    x_train = []
    y_train = []

    for i in range(look_back, len(scaled_data) - predict):
        x_train.append(scaled_data[i-look_back:i, 0]) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 0
        y_train.append(scaled_data[i:i+ predict, 0])  # To predict next value for training

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    if os.path.isdir(f'{constants.AI_DIR}/{ADVANCED_MODEL_NAME}_{look_back}_{predict}'):
        model = tf.keras.models.load_model(
            f'{constants.AI_DIR}/{ADVANCED_MODEL_NAME}_{look_back}_{predict}') # f'{constants.AI_DIR}/{SIMPLE_MODEL_NAME}_{x_train.shape[1]}'
    else:
        model = model_v2(x_train.shape[1], predict)
        model.summary()
        model.compile(optimizer='adam',
                    loss='mean_squared_error')
        
        model.fit(x_train,
                  y_train,
                  epochs=epochs)
        model.save(
            f'{constants.AI_DIR}/{ADVANCED_MODEL_NAME}_{look_back}_{predict}')
        
    return model


def predict_simple_next_values(data: pd.DataFrame, look_back: int = 100, next: int = 100):
    """
    Predict next values for some data

    Args:
        data (pandas.DataFrame)
        look_back (int): The number of previous data points to use for prediction.
        next (int): How much next values to predict.

    Returns:
        Next predicted values
    """

    if os.path.isdir(f'{constants.AI_DIR}/{SIMPLE_MODEL_NAME}_{look_back}'):
        model = tf.keras.models.load_model(
            f'{constants.AI_DIR}/{SIMPLE_MODEL_NAME}_{look_back}')
    else:
        if not query_yes_no("Model for such configuration doesn't exist. Create?", default="no"):
            return
        model = train_simple_model(data=data, look_back=look_back)
        
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

    # get the last 'look_back' values from the dataset to use as initial input
    initial_input = scaled_data[-look_back:]
    x_test = np.array([initial_input])
    predictions = []

    # predict future values
    for i in range(next):
        predicted_price = model.predict(x_test)
        predictions.append(predicted_price[0][0])

        # update input for next prediction
        initial_input = np.append(initial_input[1:], predicted_price, axis=0)
        x_test = np.array([initial_input])

    # invert scaling on predictions to get actual prices
    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1))

    return predictions


def predict_advanced_next_values(data: pd.DataFrame, look_back: int = 100, predict: int = 10):
    """
    Advanced predict next values for some data

    Args:
        data (pandas.DataFrame)
        look_back (int): The number of previous data points to use for prediction.
        predict (int): How much next values to predict.

    Returns:
        Next predicted values
    """

    if os.path.isdir(f'{constants.AI_DIR}/{ADVANCED_MODEL_NAME}_{look_back}_{predict}'):
        model = tf.keras.models.load_model(
            f'{constants.AI_DIR}/{ADVANCED_MODEL_NAME}_{look_back}_{predict}')
    else:
        if not query_yes_no("Model for such configuration doesn't exist. Create?", default="no"):
            return
        model = train_advanced_model(data=data, look_back=look_back, predict=predict)
        
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

    # get the last 'look_back' values from the dataset to use as initial input
    initial_input = scaled_data[-look_back:]
    x_test = np.array([initial_input])

    # predict future values
    predicted_values = model.predict(x_test)

    # invert scaling on predictions to get actual prices
    predictions = scaler.inverse_transform(
        np.array(predicted_values).reshape(-1, 1))

    return predictions