import datetime
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tradinga.ai_models import mape_loss, model_v2, model_v3

import tradinga.constants as constants
from tradinga.utils_helper import query_yes_no

SIMPLE_MODEL_NAME = 'simple_model'
ADVANCED_MODEL_NAME = 'advanced_model'

def make_input_data(preprocessed_data: np.ndarray, input_window: int):
    """
    Takes preprocessed (sorted and scaled) ndarray and input window and returns model ready data

    Args:
        preprocessed_data (np.ndarray)
        input_window (int)

    Returns:
        Model ready data
    """
    x_test = []
    for x in range(input_window, len(preprocessed_data)):
        x_test.append(preprocessed_data[x-input_window:x, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test


def get_evaluation(model: tf.keras.models.Sequential, x_test: np.ndarray, y_test: np.ndarray):
    # TODO: Add description
    mse, mae = model.evaluate(x_test, y_test) #, verbose=0)
    return mse, mae


def train_model(model: tf.keras.models.Sequential, x_train: np.array, y_train: np.array, epochs: int, batch_size: int = None, x_test: np.ndarray = None, y_test: np.ndarray = None, log_name: str = None):
    """
    
    Train model
    """
    # TODO: Add description
    if log_name != None:
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + log_name
    else:
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    if isinstance(x_test, np.ndarray) and isinstance(y_test, np.ndarray):
        earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
        model.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, y_test),
                    callbacks=[tensorboard_callback, earlystop])
    else:
        model.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[tensorboard_callback])


def get_simple_model(data: pd.DataFrame, look_back: int = 200, epochs: int = 50):
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
        with tf.keras.utils.custom_object_scope({'mape_loss': mape_loss}):
            model = tf.keras.models.load_model(
                f'{constants.AI_DIR}/{SIMPLE_MODEL_NAME}_{look_back}') # f'{constants.AI_DIR}/{SIMPLE_MODEL_NAME}_{x_train.shape[1]}'
    else:
        model = model_v3(x_train.shape[1])
        model.summary()

        train_model(model,x_train,y_train,epochs)
        
        model.save(
            f'{constants.AI_DIR}/{SIMPLE_MODEL_NAME}_{look_back}')
        
    return model


def get_advanced_model(data: pd.DataFrame, look_back: int = 100, predict: int = 10, epochs: int = 50):
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
        x_train.append(scaled_data[i-look_back:i, 0])
        if predict == 1:
            y_train.append(scaled_data[i, 0])
        else:
            y_train.append(scaled_data[i:i+ predict, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    if os.path.isdir(f'{constants.AI_DIR}/{ADVANCED_MODEL_NAME}_{look_back}_{predict}'):
        model = tf.keras.models.load_model(
            f'{constants.AI_DIR}/{ADVANCED_MODEL_NAME}_{look_back}_{predict}') # f'{constants.AI_DIR}/{SIMPLE_MODEL_NAME}_{x_train.shape[1]}'
    else:
        model = model_v2(x_train.shape[1], predict)
        model.summary()
        
        train_model(model,x_train,y_train,epochs)

        model.save(
            f'{constants.AI_DIR}/{ADVANCED_MODEL_NAME}_{look_back}_{predict}')
        
    return model


def predict_simple_next_values(data: pd.DataFrame, model: tf.keras.models.Sequential, look_back: int = 100, next: int = 100):
    """
    Predict next values for some data

    Args:
        data (pandas.DataFrame)
        look_back (int): The number of previous data points to use for prediction.
        next (int): How much next values to predict.

    Returns:
        Next predicted values
    """

    # if os.path.isdir(f'{constants.AI_DIR}/{SIMPLE_MODEL_NAME}_{look_back}'):
    #     model = tf.keras.models.load_model(
    #         f'{constants.AI_DIR}/{SIMPLE_MODEL_NAME}_{look_back}')
    # else:
    #     if not query_yes_no("Model for such configuration doesn't exist. Create?", default="no"):
    #         return
    #     model = get_simple_model(data=data, look_back=look_back)
    data.sort_values('time', inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

    # get the last 'look_back' values from the dataset to use as initial input
    initial_input = scaled_data[-look_back:]
    x_test = np.array([initial_input])
    predictions = []

    # predict future values
    for i in range(next):
        predicted_price = model.predict(x_test, verbose=0)
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
        model = get_advanced_model(data=data, look_back=look_back, predict=predict)
        
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


def test_simple_model(data: pd.DataFrame, model: tf.keras.models.Sequential, look_back: int):
    """
    Test simple model

    Args:
        data (pandas.DataFrame)
        model (tf.keras.models.Sequential)
        look_back (int): The number of previous data points to use for prediction.

    Returns:
        Predicted data on original data
    """
    data.sort_values('time', inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

    x_test = []
    for x in range(look_back, len(scaled_data)):
        x_test.append(scaled_data[x-look_back:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # predict future values
    predicted_values = model.predict(x_test)

    # invert scaling on predictions to get actual prices
    predictions = scaler.inverse_transform(
        np.array(predicted_values).reshape(-1, 1))

    return predictions


def test_advanced_model(data: pd.DataFrame, model: tf.keras.models.Sequential, look_back: int, predict: int):
    """
    Test advanced model

    Args:
        data (pandas.DataFrame)
        model (tf.keras.models.Sequential)
        look_back (int): The number of previous data points to use for prediction.
        predict (int): How much next values to predict.

    Returns:
        Predicted data on original data
    """
        
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))

    x_test = []
    for x in range(look_back, len(scaled_data), predict):
        x_test.append(scaled_data[x-look_back:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # predict future values
    predicted_values = model.predict(x_test)

    # invert scaling on predictions to get actual prices
    predictions = scaler.inverse_transform(
        np.array(predicted_values).reshape(-1, 1))

    return predictions