import datetime
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tradinga.ai_models import mape_loss, model_v3

import tradinga.constants as constants

SIMPLE_MODEL_NAME = 'simple_model'

scaler = MinMaxScaler(feature_range=(0, 1))
def scale_for_ai(data: pd.Series) -> np.ndarray:
    """
    Scale data for neural network

    Args:
        data (pandas.Series)

    Returns:
        scaled data (np.ndarray)

    """
    return scaler.fit_transform(data.values.reshape(-1, 1))


def scale_back(values: np.ndarray) -> np.ndarray:
    """
    Scale back values to actual size

    Args:
        values (np.ndarray)

    Returns:
        scaled back values (np.ndarray)

    """
    return scaler.inverse_transform(
        np.array(values).reshape(-1, 1))


def get_xy_arrays(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Get x_array (input) and y_array (output) for neural network training/validation

    Args:
        values (np.ndarray)
        window (input window)

    Returns:
        x_array (input), y_array (output)

    """
    # TODO: Exception when window too large.
    x_array = []
    y_array = []

    for i in range(window, len(values)):
        x_array.append(values[i-window:i, 0])
        y_array.append(values[i, 0])  # To predict next value for training

    x_array, y_array = np.array(x_array), np.array(y_array)
    x_array = np.reshape(x_array, (x_array.shape[0], x_array.shape[1], 1))

    return x_array, y_array


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

    scaled_data = scale_for_ai(data=data['close'])

    x_train, y_train = get_xy_arrays(values=scaled_data, window=look_back)

    if os.path.isdir(f'{constants.AI_DIR}/{SIMPLE_MODEL_NAME}_{look_back}'):
            model = tf.keras.models.load_model(
                f'{constants.AI_DIR}/{SIMPLE_MODEL_NAME}_{look_back}') # f'{constants.AI_DIR}/{SIMPLE_MODEL_NAME}_{x_train.shape[1]}'
    else:
        model = model_v3(x_train.shape[1])
        model.summary()

        train_model(model,x_train,y_train,epochs)
        
        model.save(
            f'{constants.AI_DIR}/{SIMPLE_MODEL_NAME}_{look_back}')
        
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

    data.sort_values('time', inplace=True)
    scaled_data = scale_for_ai(data=data['close'])

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

    return scale_back(np.array(predictions))


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
    scaled_data = scale_for_ai(data=data['close'])

    x_test = []
    for x in range(look_back, len(scaled_data)):
        x_test.append(scaled_data[x-look_back:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # predict future values
    predicted_values = model.predict(x_test)

    return scale_back(predicted_values)