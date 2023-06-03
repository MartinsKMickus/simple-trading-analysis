import datetime
import os
import warnings
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tradinga.custom_loss import direction_loss
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tqdm

from tradinga.settings import DATA_DIR
from keras.utils import custom_object_scope


MODEL_METRICS = ["mean_squared_error", "direction_sensitive_loss", "mae", "mape_loss"]


class AIManager:
    data_columns = 6
    desired_column_index = 3  # Index of the column where the predicted value should be placed
    window = 200
    one_hot_encoding_count = 0
    batch_size = 64
    epochs = 200
    use_earlystop = True
    earlystop_patience = 50
    scaler = MinMaxScaler()
    custom_scaler = False
    model = None

    # Metrics
    direction_metric = True

    def __init__(self, data_dir: str = DATA_DIR, model_name: str = '', one_hot_encoding_count: int = 0, data_min = None, data_max = None) -> None:
        self.ai_location = data_dir + "/models/" + f"MODEL_{model_name}{self.window}"
        self.one_hot_encoding_count = one_hot_encoding_count
        if isinstance(data_min, np.ndarray) and isinstance(data_max, np.ndarray):
            combined_array = np.column_stack((data_min, data_max))
            combined_array = combined_array.T
            self.scaler.fit(combined_array)
            self.custom_scaler = True

    def scale_for_ai(self, data: pd.DataFrame) -> np.ndarray:
        """
        Scale data for neural network. Column 'time' will be dropped.

        Args:
            data (pandas.Series)

        Returns:
            scaled data (np.ndarray)

        """
        without_time = np.array(data.drop("time", axis=1))
        if self.custom_scaler:
            return self.scaler.transform(without_time)
        else:
            return self.scaler.fit_transform(without_time)
        

    def scale_back_array(self, values: np.ndarray):
        """
        Scale back values to actual size

        Args:
            values (np.ndarray)

        Returns:
            scaled back values (np.ndarray)

        """
        return self.scaler.inverse_transform(values)
    
    def scale_back_value(self, value):
        """
        Scale back 'close' value to actual size

        Args:
            value

        Returns:
            scaled back value

        """
        temp_array = np.zeros(self.data_columns)
        temp_array[self.desired_column_index] = value
        # Reshape the array to match the expected shape for inverse transformation
        temp_array = temp_array.reshape(1, -1)
        inverse_transformed_array = self.scale_back_array(temp_array)
        final_predicted_value = inverse_transformed_array[:, self.desired_column_index]
        return final_predicted_value

    def get_xy_arrays(
        self, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get x_array (input) and y_array (output) for neural network training/validation

        Args:
            values (np.ndarray)
            window (input window)

        Returns:
            x_array (input), y_array (output)

        """
        if self.window >= len(values) - 1:
            raise Exception(f"Not enough values: {len(values)} for window of: {self.window}")

        x_array = []
        y_array = []

        for i in range(self.window, len(values) - 1):
            x_array.append(
                values[i - self.window : i]
            )
            y_array.append(values[i, 3])  # To predict next 'close' value

        x_array, y_array = np.array(x_array), np.array(y_array)

        return x_array, y_array

    def get_one_hot_encoding(self, values: np.ndarray, one_hot_encoding: int = 0):
        if self.one_hot_encoding_count < one_hot_encoding:
            raise Exception(f'One hot encoding not possible. Provided index {one_hot_encoding} but saved index count is: {self.one_hot_encoding_count}')
        # One hot encoding + 1 because 0 index will stand for unknown stock
        symbol_encoded = tf.one_hot(one_hot_encoding, depth=self.one_hot_encoding_count)
        symbol_encoded_np = np.array(symbol_encoded) # Convert to NumPy array
        x_new = np.expand_dims(values, axis=tuple(range(-self.one_hot_encoding_count, 0)))
        # reshaped_encoding = symbol_encoded_np[:, np.newaxis]
        return values + symbol_encoded_np #reshaped_encoding

    def model_structure(self, i_shape, output=1):
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.LSTM(
                units=64, return_sequences=True, input_shape=(i_shape)
            )
        )
        model.add(tf.keras.layers.LSTM(units=128))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.Reshape((1, 512)))
        model.add(tf.keras.layers.LSTM(units=8))
        model.add(tf.keras.layers.Dense(16))
        model.add(tf.keras.layers.Dense(units=output))
        model.compile(optimizer="adam", loss=direction_loss, metrics=["mean_squared_error", "mae"])
        model.summary()
        return model

    def load_model(self):
        # TODO: Add description
        if os.path.exists(self.ai_location):
            with custom_object_scope({'direction_loss': direction_loss}):
                self.model = tf.keras.models.load_model(self.ai_location)

    def save_model(self):
        # TODO: Add description
        if isinstance(self.model, tf.keras.Model):
            self.model.save(self.ai_location)
        else:
            print("Save model called but model does not exist!")

    def create_model(self, shape):
        # TODO: Add description
        self.model = self.model_structure(i_shape=shape)

    def get_evaluation(self, model: tf.keras.models.Sequential, x_test: np.ndarray, y_test: np.ndarray):
        # TODO: Add description
        return model.evaluate(x_test, y_test) #, verbose=0)

    def train_model(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test=None,
        y_test=None,
        log_name=None,
    ):
        """
        Train model on given arrays. 

        Args:
            x_train (np.ndarray)
            y_train (np.ndarray)
            x_test (np.ndarray)
            y_test (np.ndarray)
            log_name (str) train log name

        """
        # TODO: Add description
        if not isinstance(self.model, tf.keras.Model):
            raise Exception('Train model called but there is no model loaded')

        if isinstance(log_name, str):
            log_dir = (
                "logs/fit/"
                + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                + log_name
            )
        else:
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1, write_images=False
        )
        # if isinstance(one_hot_encoding, np.ndarray):
        #     x_test = [x_test, one_hot_encoding]

        if isinstance(x_test, np.ndarray) and isinstance(y_test, np.ndarray):
            if self.use_earlystop:
                earlystop = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.earlystop_patience,
                    verbose=1,
                    mode="auto",
                )
                self.model.fit(
                    x_train,
                    y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_data=(x_test, y_test),
                    callbacks=[tensorboard_callback, earlystop],
                )
            else:
                self.model.fit(
                    x_train,
                    y_train,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    validation_data=(x_test, y_test),
                    callbacks=[tensorboard_callback],
                )
        else:
            self.model.fit(
                x_train,
                y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[tensorboard_callback],
            )

    def predict_next_value(self, values: np.ndarray) -> int:
        if not isinstance(self.model, tf.keras.Model):
            print("Predict called but model does not exist!")
            raise Exception("No model trained")
        
        input = values[-self.window:]
        input = np.expand_dims(values[-self.window:], axis=0)
        # predict future values
        predicted = self.model.predict(input, verbose='0')
        
        return predicted[0][0]

    def get_metrics_on_data(self, values: np.ndarray, symbol: str = ''):
        correct_direction_count = 0
        value_count = len(values) - 1

        for i in tqdm.tqdm(range(self.window, len(values) - 1), desc=f'Calculating metrics {symbol}'):
            predicted = self.predict_next_value(values[i - self.window : i])
            actual_value = values[i, 3]
            previous_value = values[i-1, 3]
            if self.direction_metric:
                if predicted > previous_value and actual_value > previous_value:
                    correct_direction_count += 1
                elif predicted < previous_value and actual_value < previous_value:
                    correct_direction_count += 1

        return correct_direction_count/(len(values) - 1 - self.window)