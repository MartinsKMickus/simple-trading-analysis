

import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tradinga import constants

from tradinga.ai_helper import get_evaluation, model_v3, train_model


def make_few_models(data: pd.DataFrame, test_data: pd.DataFrame, test_name: str):
    
    look_back = 100
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
    
    # if os.path.isdir(f'{constants.AI_DIR}/{test_name}'):
    #     print("Found such test. Skipping.")
    #     return
    accuracy = 0
    accuracies = []
    best_model = ""
    for start_units in range(100, 161, 20):
        for dense_after in range(50, 71, 10):
            if not os.path.isdir(f'{constants.AI_DIR}/{test_name}/model_SU{start_units}_DA{dense_after}'):
                model = model_v3(x_train.shape[1], first_units=start_units, dense_after=dense_after)
                train_model(model,x_train,y_train,look_back)
                model.save(
                    f'{constants.AI_DIR}/{test_name}/model_SU{start_units}_DA{dense_after}')
            else:
                model = tf.keras.models.load_model(f'{constants.AI_DIR}/{test_name}/model_SU{start_units}_DA{dense_after}')
            
            # current_accuracy = get_evaluation(model, x_test, y_test)
            # accuracies.append(current_accuracy)
            # if accuracy < current_accuracy:
            #     accuracy = current_accuracy
            #     best_model = f"Best model is model_SU{start_units}_DA{dense_after}"
    
    # print(best_model)
    # plt.plot(accuracies)
    # plt.show()
    
