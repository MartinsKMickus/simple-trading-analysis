import datetime
import os
import signal
import sys
import warnings
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tradinga.custom_loss import direction_loss
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

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
    valuation_metrics = ['Correct trend loss']

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
        if isinstance(value, np.ndarray):
            temp_array = np.zeros(shape=(value.shape[0], self.data_columns))
            value = np.squeeze(value)
            temp_array[:, self.desired_column_index] = value
        else:
            temp_array = np.zeros(shape=self.data_columns)
            temp_array[self.desired_column_index] = value
            temp_array = temp_array.reshape(1, -1)
        # Reshape the array to match the expected shape for inverse transformation

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
            y_array.append(values[i, self.desired_column_index])  # To predict next 'close' value

        x_array, y_array = np.array(x_array), np.array(y_array)

        return x_array, y_array

    # One hot encoding
    def get_one_hot_encoding(self, one_hot_encoding: int = 0):
        if self.one_hot_encoding_count < one_hot_encoding:
            raise Exception(f'One hot encoding not possible. Provided index {one_hot_encoding} but saved index count is: {self.one_hot_encoding_count}')
        # One hot encoding + 1 because 0 index will stand for unknown stock
        symbol_encoded = tf.one_hot(one_hot_encoding, depth=self.one_hot_encoding_count)
        symbol_encoded_np = np.array(symbol_encoded) # Convert to NumPy array
        # x_new = np.expand_dims(values, axis=tuple(range(-self.one_hot_encoding_count, 0)))
        # symbol_encoded_np = symbol_encoded_np[:, np.newaxis]
        return symbol_encoded_np #reshaped_encoding

    def model_structure(self, i_shape, output=1):
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.LSTM(
                units=128, return_sequences=True, input_shape=(i_shape)
            )
        )
        model.add(tf.keras.layers.LSTM(units=256))
        model.add(tf.keras.layers.Dense(1024))
        # model.add(tf.keras.layers.Reshape((1, 2048)))
        # model.add(tf.keras.layers.LSTM(units=64))
        # model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.Dense(512))
        model.add(tf.keras.layers.Dense(units=output))
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=[direction_loss, "mae"])
        model.summary()
        return model
    
    # One hot encoding
    def model_structure2(self, data_shape, category_shape, output=1):
        # Define the input layers
        existing_input = tf.keras.Input(shape=data_shape)
        categorical_input = tf.keras.Input(shape=category_shape)
        categorical_input = tf.keras.layers.Reshape((1, 5254))(categorical_input)
        model = tf.keras.models.Sequential()
        x = tf.keras.layers.LSTM(units=64, return_sequences=True)(existing_input)
        x = tf.keras.layers.LSTM(units=128)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.Reshape((1, 512))(x)
        # Concatenate the existing_input and categorical_input
        x = tf.keras.layers.Concatenate()([x, categorical_input])
        # x = tf.keras.layers.Concatenate()([x, tf.keras.layers.Reshape((1, self.one_hot_encoding_count))(categorical_input)])
        x = tf.keras.layers.LSTM(units=8)(x)
        x = tf.keras.layers.Dense(16)(x)
        output = tf.keras.layers.Dense(units=output)(x)
        model = tf.keras.Model(inputs=[existing_input, categorical_input], outputs=output)
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

    # One hot encoding
    def create_model2(self, data_shape, category_shape):
        # TODO: Add description
        self.model = self.model_structure2(data_shape=data_shape, category_shape=category_shape)

    def get_evaluation(self, x_test: np.ndarray, y_test: np.ndarray) -> list:
        # TODO: Add description
        if isinstance(self.model, tf.keras.Model):
            return list(self.model.evaluate(x_test, y_test, verbose='0'))
        return [0]

    def train_model(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        one_hot=None, # One hot encoding
        one_hot_test=None, # One hot encoding
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
            log_dir=log_dir, histogram_freq=1, write_images=True
        )
        # if isinstance(one_hot_encoding, np.ndarray):
        #     x_test = [x_test, one_hot_encoding]
        if isinstance(one_hot, np.ndarray):
            one_hot = np.expand_dims(one_hot, axis=0)  # Expand dimensions along the samples axis
            one_hot = np.repeat(one_hot, len(x_train), axis=0)  # Repeat the array for each sample
            # one_hot = np.expand_dims(one_hot[-self.window:], axis=0)

        if isinstance(one_hot_test, np.ndarray):
            one_hot_test = np.expand_dims(one_hot_test, axis=0)  # Expand dimensions along the samples axis
            one_hot_test = np.repeat(one_hot_test, len(x_train), axis=0)  # Repeat the array for each sample
            # one_hot_test = np.expand_dims(one_hot_test[-self.window:], axis=0)

        if isinstance(x_test, np.ndarray) and isinstance(y_test, np.ndarray):
            if self.use_earlystop:
                earlystop = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.earlystop_patience,
                    verbose=1,
                    mode="auto",
                )
                for _ in tqdm.tqdm(range(self.epochs), unit='epoch', desc='Progress'):
                    self.model.fit(
                        x_train,
                        y_train,
                        epochs=1,
                        batch_size=self.batch_size,
                        validation_data=(x_test, y_test),
                        callbacks=[tensorboard_callback, earlystop],
                        verbose='0',
                    )
            else:
                for _ in tqdm.tqdm(range(self.epochs), unit='epoch', desc='Progress'):
                    self.model.fit(
                        x_train,
                        y_train,
                        epochs=1,
                        batch_size=self.batch_size,
                        validation_data=(x_test, y_test),
                        callbacks=[tensorboard_callback],
                        verbose='0',
                    )
        else:
            for _ in tqdm.tqdm(range(self.epochs), unit='epoch', desc='Progress'):
                self.model.fit(
                    x_train,
                    y_train,
                    epochs=1,
                    batch_size=self.batch_size,
                    callbacks=[tensorboard_callback],
                    verbose='0',
                )

    def predict_next_value(self, values: np.ndarray, one_hot_encoding = None) -> int:
        if not isinstance(self.model, tf.keras.Model):
            print("Predict called but model does not exist!")
            raise Exception("No model loaded")
        
        input = values[-self.window:]
        input = np.expand_dims(values[-self.window:], axis=0)
        # predict future values
        # One hot encoding
        if isinstance(one_hot_encoding, np.ndarray):
            predicted = self.model.predict([input, one_hot_encoding], verbose='0')
        else:
            predicted = self.model.predict(input, verbose='0')
        
        return predicted[0][0]
    
    def predict_all_values(self, values: np.ndarray, one_hot_encoding = None) -> np.ndarray:
        if not isinstance(self.model, tf.keras.Model):
            print("Predict called but model does not exist!")
            raise Exception("No model loaded")
        
        x_array = []
        for x in range(self.window, len(values)):
            x_array.append(values[x-self.window:x])

        values = np.array(x_array)
        # values = np.reshape(x_array, (x_array.shape[0], x_array.shape[1], 1))

        if isinstance(one_hot_encoding, np.ndarray):
            predicted = self.model.predict([values, one_hot_encoding], verbose='0')
        else:
            predicted = self.model.predict(values, verbose='0')

        return predicted

    def get_metrics_on_data(self, values: np.ndarray, symbol: str = '', one_hot_encoding = None) -> list[float]:
        correct_direction_count = 0
        # - 1 Because last value is for comparison only
        value_count = len(values) - 1
        # max_change = 0.3
        # print(f'Required change: {max_change}')

        try:
            for i in tqdm.tqdm(range(self.window, len(values) - 1), desc=f'Calculating metrics {symbol}'):
                # One hot encoding
                if isinstance(one_hot_encoding, np.ndarray):
                    predicted = self.predict_next_value(values[i - self.window : i], one_hot_encoding=one_hot_encoding)
                else:
                    predicted = self.predict_next_value(values[i - self.window : i])
                actual_value = self.scale_back_value(values[i, 3])
                previous_value = self.scale_back_value(values[i-1, 3])
                predicted = self.scale_back_value(predicted)
                if self.direction_metric:
                    # Check precision only if predicted change is large enough
                    # if abs((predicted-previous_value)/previous_value) > max_change:
                    #     value_count -= 1
                    #     continue
                    if predicted > previous_value and actual_value > previous_value:
                        correct_direction_count += 1
                    elif predicted < previous_value and actual_value < previous_value:
                        correct_direction_count += 1

        except KeyboardInterrupt:
            print("Ctrl+C detected. Stopping the program...")
            # Perform any necessary cleanup or finalization steps
            sys.exit(0)

        if value_count - self.window == 0:
            return [0]
        return [correct_direction_count/(value_count - self.window)]