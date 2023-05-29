


import datetime
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from tradinga.settings import DATA_DIR
from keras.utils import custom_object_scope


MODEL_METRICS = [ 'mean_squared_error', 'direction_sensitive_loss','mae', 'mape_loss']

def model_v3(i_shape, output = 1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=64, return_sequences=True, input_shape=(i_shape, 1)))
    model.add(tf.keras.layers.LSTM(units=16))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Reshape((1, 128)))
    model.add(tf.keras.layers.LSTM(units=8, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=4))
    model.add(tf.keras.layers.Dense(4))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(units=output))
    model.compile(optimizer='adam',
                loss='mean_squared_error', metrics=['mae'])
    model.summary()
    return model

def load_model(path: str):
    # with custom_object_scope({'direction_sensitive_loss': direction_sensitive_loss, 'mape_loss': mape_loss}):
    return tf.keras.models.load_model(path)
    
    
class AIManager:

    window = 200
    batch_size = 64
    use_earlystop = False
    earlystop_patience = 50
    scaler = MinMaxScaler()
    model = None

    def __init__(self, data_dir: str = DATA_DIR) -> None:
        ai_location = data_dir + "/" + f'MODEL_{self.window}'
        if os.path.isdir(ai_location):
            self.model = load_model(ai_location)
    
    def scale_for_ai(self, data: pd.DataFrame) -> np.ndarray:
        """
        Scale data for neural network

        Args:
            data (pandas.Series)

        Returns:
            scaled data (np.ndarray)

        """
        return self.scaler.fit_transform(data.values)


    def scale_back(self, values: np.ndarray) -> np.ndarray:
        """
        Scale back values to actual size

        Args:
            values (np.ndarray)

        Returns:
            scaled back values (np.ndarray)

        """
        return self.scaler.inverse_transform(values)

    def get_xy_arrays(self, values: np.ndarray, window: int = window) -> tuple[np.ndarray, np.ndarray]:
        """
        Get x_array (input) and y_array (output) for neural network training/validation

        Args:
            values (np.ndarray)
            window (input window)

        Returns:
            x_array (input), y_array (output)

        """
        x_array = []
        y_array = []

        for i in range(window, len(values)):
            x_array.append(values[i-window:i, [1, 2, 3, 4, 5]])  # Include columns 1, 2, 3, 4 (volume, open, high, low)
            y_array.append(values[i, 0])  # To predict next 'close' value

        x_array, y_array = np.array(x_array), np.array(y_array)
        x_array = np.reshape(x_array, (x_array.shape[0], x_array.shape[1], 1))

        return x_array, y_array
    
    def train_model(self, model: tf.keras.models.Sequential, x_train: np.ndarray, y_train: np.ndarray, epochs: int, x_test = None, y_test = None, log_name = None):
        """
        
        Train model
        """
        # TODO: Add description
        if isinstance(log_name, str):
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + log_name
        else:
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True)
        
        if isinstance(x_test, np.ndarray) and isinstance(y_test, np.ndarray):
            if self.use_earlystop:
                earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.earlystop_patience, verbose=1, mode='auto')
                model.fit(x_train,
                            y_train,
                            epochs=epochs,
                            batch_size=self.batch_size,
                            validation_data=(x_test, y_test),
                            callbacks=[tensorboard_callback, earlystop])
            else:
                model.fit(x_train,
                            y_train,
                            epochs=epochs,
                            batch_size=self.batch_size,
                            validation_data=(x_test, y_test),
                            callbacks=[tensorboard_callback])
        else:
            model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=self.batch_size,
                        callbacks=[tensorboard_callback])