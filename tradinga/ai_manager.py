import datetime
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tradinga import settings

from tradinga.custom_loss import IntervalAccuracy, direction_loss
from tradinga.tools import bcolors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow as tf
import tqdm

from tradinga.settings import DATA_DIR
from keras.utils import custom_object_scope


MODEL_METRICS = ["mean_squared_error", "direction_sensitive_loss", "mae", "mape_loss"]

class AIManager:
    data_columns = 1 # DO NOT CHANGE!!!
    desired_column_index = 0  # Index of the column where the predicted value should be placed
    window = 5 # DO NOT CHANGE!!!
    one_hot_encoding_count = 0
    batch_size = 64
    epochs = settings.SINGLE_DATA_EPOCHS
    use_earlystop = False
    earlystop_patience = 50
    scaler = MinMaxScaler()
    custom_scaler = False
    model = None
    # After how much time units make prediction in future
    predict_after_time = settings.PREDICT_AFTER #round(10/(3/5)) #10 # 7 hours in one working day.

    # Metrics
    direction_metric = True
    valuation_metrics = ['Correct trend loss']

    # Monte Carlo model confidence (variance in percentage).
    monte_carlo_samples = 100
    diversity_threshold = settings.DIVERSITY_THRESHOLD # This is how much toward last value take profit has to be moved
    use_std = True # 
    batch_memory = 1024

    # Required model confidence to base metric score on. Model could be wrong in most cases but with low confidence.
    # We'll assume model should get better with confidence scores over 50%
    required_confidence = settings.MIN_CONFIDENCE

    def __init__(self, data_dir: str = DATA_DIR, model_name: str = '', one_hot_encoding_count: int = 0, data_min = None, data_max = None, window: int = 5) -> None:
        self.window = window
        self.ai_location = data_dir + "/models/" + f"MODEL_{model_name}{self.window}"
        self.one_hot_encoding_count = one_hot_encoding_count
        # For calculations next day is 0 and so on
        self.predict_after_time -= 1
        if isinstance(data_min, np.ndarray) and isinstance(data_max, np.ndarray):
            self.apply_minmax_setting(data_min=data_min, data_max=data_max)

    def apply_minmax_setting(self, data_min: np.ndarray, data_max: np.ndarray):
        """
        Applies scaler configuration

        Args:
            data_min (np.ndarray): Data min
            data_max (np.ndarray): Data max

        """
        combined_array = np.column_stack((data_min, data_max))
        combined_array = combined_array.T
        self.scaler.fit(combined_array)
        self.custom_scaler = True
        
    def scale_for_ai(self, data: pd.DataFrame) -> np.ndarray:
        """
        Scale data for neural network. Column 'time' will be dropped.

        Args:
            data (pandas.Series): In form of (time,open,high,low,close,adj close,volume)

        Returns:
            scaled data (np.ndarray)

        """
        without_time = np.array(data.drop("time", axis=1))
        # print(data.columns)
        if self.custom_scaler:
            return self.scaler.transform(without_time)
        else:
            return self.scaler.fit_transform(without_time)
        

    def scale_back_array(self, values: np.ndarray):
        """
        Scale back values to actual size.

        Args:
            values (np.ndarray)

        Returns:
            scaled back values (np.ndarray)

        """
        return self.scaler.inverse_transform(values)
    
    def scale_back_value(self, value):
        """
        Scale back 'close' value to actual size. Can scale back only one value and also multiple values.

        Args:
            value (np.ndarray) or (float): Values between 0 and 1.

        Returns:
            Scaled back value or values.

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

        for i in range(self.window, len(values) - self.predict_after_time):
            x_array.append(
                values[i - self.window : i]
            )
            y_array.append(values[i + self.predict_after_time, self.desired_column_index])  # To predict 'close' value
            # if values[i + self.predict_after_time, self.desired_column_index] > values[i - 1, self.desired_column_index]:
            #     y_array.append(1)
            # else:
            #     y_array.append(0)

        x_array, y_array = np.array(x_array), np.array(y_array)

        return x_array, y_array

    # One hot encoding
    def get_one_hot_encoding(self, one_hot_encoding: int = 0):
        """
        UNDER DEVELOPMENT! Gets one hot encoding for symbol ID to provide to model.
        Args:
            one_hot_encoding (int): Stock ID to encode

        Returns:
            Encoded array
        """
        if self.one_hot_encoding_count < one_hot_encoding:
            raise Exception(f'One hot encoding not possible. Provided index {one_hot_encoding} but saved index count is: {self.one_hot_encoding_count}')
        # One hot encoding + 1 because 0 index will stand for unknown stock
        symbol_encoded = tf.one_hot(one_hot_encoding, depth=self.one_hot_encoding_count)
        symbol_encoded_np = np.array(symbol_encoded) # Convert to NumPy array
        # x_new = np.expand_dims(values, axis=tuple(range(-self.one_hot_encoding_count, 0)))
        # symbol_encoded_np = symbol_encoded_np[:, np.newaxis]
        return symbol_encoded_np #reshaped_encoding

    def model_structure(self, i_shape=(200, 6), output=1):
        """
        Creates default model with provided input shape and output value count.

        Args:
            i_shape (tuple): Model input shape.
            output (int): Model output count.

        Returns:
            Model (tf.keras.models.Sequential)

        """
        print(f'Model input shape: {i_shape}')
        model = tf.keras.models.Sequential()
        # model.add(tf.keras.layers.LSTM(units=16, return_sequences=True, input_shape=i_shape)) # It is important to capture order rather than values at first
        model.add(tf.keras.layers.Dense(512, activation='elu', input_shape=i_shape)) # Dense layer provides individual value importance after LSTM
        model.add(tf.keras.layers.Dense(512, activation='elu'))
        model.add(tf.keras.layers.Dense(512, activation='elu'))
        model.add(tf.keras.layers.Dense(512, activation='elu'))
        # model.add(tf.keras.layers.Dropout(0.1))
        # model.add(tf.keras.layers.Dense(64, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
        # model.add(tf.keras.layers.Dense(16))
        
        # model.add(tf.keras.layers.LSTM(units=8))
        # model.add(tf.keras.layers.Dense(8))
        # model.add(tf.keras.layers.Dropout(0.01))
        # model.add(tf.keras.layers.LSTM(units=4))
        # model.add(tf.keras.layers.Dense(4))
        # model.add(tf.keras.layers.Reshape((1, 300)))
        model.add(tf.keras.layers.GlobalAveragePooling1D())
        # model.add(tf.keras.layers.Dense(8))
        model.add(tf.keras.layers.Dense(units=output)) # , activation='sigmoid'
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        edited_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) # RMSprop, Adagrad, SGD, Adam
        model.compile(optimizer=edited_optimizer, loss="mae", metrics=["mean_squared_error", direction_loss, IntervalAccuracy()])
        model.summary()
        return model
    
    # One hot encoding
    def model_structure2(self, data_shape, category_shape, output=1):
        """
        Creates model with one hot encoding capability, provided input shape and output value count.

        Args:
            i_shape (tuple): Model input shape.
            category_shape (tuple): One hot encoding shape.
            output (int): Model output count.

        Returns:
            Model (tf.keras.models.Sequential)

        """
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

    # def model_structure_features(self, i_shapes=(200, 6), output=1):
    #     # Features as inputs
    #     input1 = tf.keras.layers.Input(shape=(200,))
    #     input2 = tf.keras.layers.Input(shape=(200,))


    #     # Process the first input in its own layer
    #     layer1_output = tf.keras.layers.Dense(units=64, activation='relu')(input1)
    #     # Process the second input in its own layer
    #     layer2_output = tf.keras.layers.Dense(units=32, activation='relu')(input2)

    #     # Concatenate the outputs from the two layers
    #     concatenated = tf.keras.layers.concatenate([layer1_output, layer2_output])

    #     # Continue with additional layers as needed
    #     output = tf.keras.layers.Dense(units=1, activation='sigmoid')(concatenated)

    #     # Create the model
    #     model = tf.keras.models.Model(inputs=[input1, input2], outputs=output)
    #     return model

    def load_model(self):
        """
        Loads model in class variable using custom object scope to be able to work with custom loss functions/

        Returns:
            Model (tf.keras.model)

        """
        if os.path.exists(self.ai_location):
            with custom_object_scope({'direction_loss': direction_loss, 'IntervalAccuracy': IntervalAccuracy}):
                self.model = tf.keras.models.load_model(self.ai_location)

    def save_model(self):
        """
        Saves model stored in class variable. Overwrites existing one if present.

        """
        if isinstance(self.model, tf.keras.Model):
            self.model.save(self.ai_location)
        else:
            print("Save model called but model does not exist!")

    def create_model(self, shape):
        """
        Creates new model and overwrites class model variable.

        """
        self.model = self.model_structure(i_shape=shape)

    # One hot encoding
    def create_model2(self, data_shape, category_shape):
        """
        Creates new model with one hot encoding capability and overwrites class model variable.

        """
        self.model = self.model_structure2(data_shape=data_shape, category_shape=category_shape)

    def get_evaluation(self, x_array: np.ndarray, y_array: np.ndarray) -> list:
        """
        Gets evaluation metrics from model using x_array and y_array

        Args:
            x_array (np.ndarray): X array
            y_array (np.ndarray): Y array

        Returns:
            Loss function metric value

        """
        if isinstance(self.model, tf.keras.Model):
            return list(self.model.evaluate(x_array, y_array, verbose='0'))
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
        Train model on given arrays. Including logging.

        Args:
            x_train (np.ndarray)
            y_train (np.ndarray)
            x_test (np.ndarray)
            y_test (np.ndarray)
            log_name (str) train log name

        """
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

    def predict_next_value(self, values: np.ndarray, one_hot_encoding = None) -> tuple[float, float]:
        """
        Gets next (predicted) value for scaled values and confidence.

        Args:
            values (np.ndarray): Scaled values.
            one_hot_encoding: UNDER DEVELOPMENT

        Returns:
            (Predicted value, confidence)

        """
        if not isinstance(self.model, tf.keras.Model):
            print("Predict called but model does not exist!")
            raise Exception("No model loaded")
        
        input_values = values[-self.window:]
        input_values = np.expand_dims(input_values, axis=0)
        
        # Monte Carlo Dropout predictions with confidence estimates
        predictions = np.zeros((self.monte_carlo_samples, 1))

        # import tensorflow.keras.backend as K
        # K.set_learning_phase(1)
        for i in range(self.monte_carlo_samples):
            predictions[i] = self.model(input_values, training=True)

        # Compute mean and standard deviation for each prediction
        mean_prediction = np.mean(predictions)
        std_prediction = np.std(predictions)

        # Implementation of #31 had no correct assumption. Code is cleaned up instead.
        
        if std_prediction == 0:
            confidence = 1.0  # All predicted values are the same
            # print(f'{bcolors.BOLD}{bcolors.OKGREEN}IMPOSSIBLE HAS HAPPENED!!! CONFIDENCE 100%{bcolors.ENDC}')
        elif std_prediction >= self.diversity_threshold:
            confidence = 0.0  # Diversity exceeds or equals the threshold
        else:
            diversity = np.max(predictions) - np.min(predictions)
            if self.use_std:
                confidence = max(1.0 - float(std_prediction / self.diversity_threshold), 0.0)
            else:
                confidence = max(1.0 - float(diversity / self.diversity_threshold), 0.0)

        prediction = float(mean_prediction)
        return prediction, confidence
    
    def predict_all_values(self, values: np.ndarray, one_hot_encoding = None, monte_carlo: bool = False):
        """
        Gets all values that can be predicted within provided data amount from window size -> end of data.

        Args:
            values (np.ndarray): Scaled values.
            one_hot_encoding: UNDER DEVELOPMENT

        Returns:
            Predicted value.

        """
        if not isinstance(self.model, tf.keras.Model):
            print("Predict called but model does not exist!")
            raise Exception("No model loaded")
        
        x_array = []
        for x in range(self.window, len(values)):
            x_array.append(values[x-self.window:x])

        values = np.array(x_array)

        if monte_carlo:
            # Monte Carlo Dropout predictions with confidence estimates
            predictions = np.zeros((self.monte_carlo_samples, len(values)))

            # tf.keras.backend.set_learning_phase(1)
            for i in tqdm.tqdm(range(self.monte_carlo_samples), delay=10, desc='Monte Carlo dropout calculations'):
                for batch_pred in range(0, len(values), self.batch_memory):
                    if batch_pred + self.batch_memory > len(values):
                        max_border = len(values)
                    else:
                        max_border = batch_pred + self.batch_memory

                    predictions[i, batch_pred:max_border] = np.array(self.model(values[batch_pred:max_border], training=True)).flatten() #predict(values).flatten() # , training=True

            # Compute mean and standard deviation for each prediction
            mean_predictions = np.mean(predictions, axis=0)
            std_predictions = np.std(predictions, axis=0)
            confidence_scores = []
            # Implementation of #31 had no correct assumption. Code is cleaned up instead.
            for i in range(len(std_predictions)):
                if std_predictions[i] == 0:
                    confidence = 1.0  # All predicted values are the same
                    # print(f'{bcolors.BOLD}{bcolors.OKGREEN}IMPOSSIBLE HAS HAPPENED!!! CONFIDENCE 100%{bcolors.ENDC}')
                elif std_predictions[i] >= self.diversity_threshold:
                    confidence = 0.0  # Diversity exceeds or equals the threshold
                else:
                    if self.use_std:
                        confidence = max(1.0 - float(std_predictions[i] / self.diversity_threshold), 0.0)
                    else:
                        diversity = np.max(predictions[:,i]) - np.min(predictions[:,i])
                        confidence = max(1.0 - float(diversity / self.diversity_threshold), 0.0)
                confidence_scores.append(confidence)
                
            return mean_predictions, confidence_scores
        else:
            if isinstance(one_hot_encoding, np.ndarray):
                predicted = self.model.predict([values, one_hot_encoding], verbose='0')
            else:
                predicted = np.array(self.model(values)) # , verbose='0'

            return predicted

    def get_metrics_on_data(self, values: np.ndarray, symbol: str = '', one_hot_encoding = None) -> list[float]:
        """
        Gets metrics for model precision on data

        Args:
            values (np.ndarray): Scaled values.
            symbol (str): Can be provided for progress bar information.
            one_hot_encoding: UNDER DEVELOPMENT

        Returns:
            Currently one metric (precision on predicting trend up/down)

        """
        correct_direction_count = 0
        predictions, confidences = self.predict_all_values(values=values, one_hot_encoding=one_hot_encoding, monte_carlo=True)
        predictions = self.scale_back_value(predictions)
        # We need to check exact values of passing confidences to calculate average.
        # If all predictions was right then confidence should be also 100% but if it is not then metric cannot exceed average of these confidences (to make model provide correct confidence)
        # In other cases metric could be just accuracy of predictions
        checked_confidences = []
        # TODO: IMPLEMENT BOOL METRIC rather that precise value prediction
        # predictions_bool = self.predict_all_values(values=values, one_hot_encoding=one_hot_encoding)
        values = self.scale_back_value(value=values[:,self.desired_column_index])
        try:
            for i in range(1 + self.predict_after_time, len(predictions) - self.predict_after_time):
                predicted = predictions[i]
                if confidences[i] < self.required_confidence:
                    continue
                checked_confidences.append(confidences[i])
                actual_value = values[i + self.window + self.predict_after_time]
                previous_value = values[i + self.window - 1]
                previous_predicted = predictions[i - 1 - self.predict_after_time]
                if self.direction_metric:
                    # Based on previous
                    # if predicted > previous_value and actual_value > previous_value:
                    #     correct_direction_count += 1
                    # elif predicted < previous_value and actual_value < previous_value:
                    #     correct_direction_count += 1
                    # Based on predictions
                    if predicted > previous_predicted and actual_value > previous_value:
                        correct_direction_count += 1
                    elif predicted < previous_predicted and actual_value < previous_value:
                        correct_direction_count += 1

                    # Bool
                    # if actual_value > previous_value and predicted > 0.5:
                    #     correct_direction_count += 1
                    # elif actual_value < previous_value and predicted < 0.5:
                    #     correct_direction_count += 1

        except KeyboardInterrupt:
            print("Ctrl+C detected. Stopping the program...")
            # Perform any necessary cleanup or finalization steps
            sys.exit(0)

        if len(checked_confidences) == 0:
            # There were no predictions above defined confidence. In this case accuracy should be 0 for this stock
            return [0]
        
        accuracy = correct_direction_count/len(checked_confidences)
        conf_average = sum(checked_confidences)/len(checked_confidences)
        if accuracy > conf_average:
            accuracy = conf_average

        return [accuracy]
    
    def get_learning_setting_summary(self):
        """
        Prints learning setting summary.
        """
        if not isinstance(self.model, tf.keras.Model):
            print("Learning setting summary called but model does not exist!")
            return
        print(f'{bcolors.BOLD}Model Learning Setting Summary:{bcolors.ENDC}')
        print(f'Input Window: {self.window}')
        print(f'Predict after: {self.predict_after_time + 1}')
        print(f'Single Data Epochs: {self.epochs}')
        print(f'Accuracy Cutoff: {settings.ACCURACY_CUTOFF}')
        print(f'Model Loss Function: {self.model.loss}')
        print(f'Model Learning Rate: {self.model.optimizer.learning_rate.numpy()}')
        print(f'Min Confidence: {settings.MIN_CONFIDENCE}')
        print(f'Diversity Threshold: {settings.DIVERSITY_THRESHOLD}')
        try:
            input("Press Enter to continue!\n")
        except KeyboardInterrupt:
            print("Ctrl+C detected. Stopping the program...")
            sys.exit(0)