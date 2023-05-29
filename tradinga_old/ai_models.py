import tensorflow as tf
import keras.backend as K
from keras.utils import custom_object_scope


MODEL_METRICS = [ 'mean_squared_error', 'direction_sensitive_loss','mae', 'mape_loss']


def mape_loss(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
    return 100. * K.mean(diff, axis=-1)
    

# def direction_sensitive_loss(y_true, y_pred):
#     # Shift y_true and y_pred by one step to get the previous values
#     prev_true = tf.concat([y_true[-1:], y_true[:-1]], axis=0)
#     prev_pred = tf.concat([y_pred[-1:], y_pred[:-1]], axis=0)
    
#     # Compute the difference between the current and previous values
#     diff = y_true - prev_pred
    
#     # Compute the direction of the next real value
#     direction = tf.sign(diff)
    
#     # Compute the direction of the predicted value
#     pred_direction = tf.sign(y_pred - prev_true)
    
#     # Compute the loss based on the difference and the direction
#     loss = tf.where(tf.equal(direction, pred_direction), tf.abs(diff), 2 * tf.abs(diff))
    
#     # Take the average over the sequence
#     return tf.reduce_mean(loss)


def direction_sensitive_loss(y_true, y_pred):
    prev_true = tf.concat([y_true[-1:], y_true[:-1]], axis=0)
    direction = tf.sign(y_true - prev_true)
    pred_direction = tf.sign(y_pred - prev_true)
    loss = tf.abs(direction - pred_direction) * 0.5 # * tf.abs(y_true - y_pred) #* 0.5  # Scale the loss values
    return tf.reduce_mean(loss)


def model_v3(i_shape, output = 1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=64, return_sequences=True, input_shape=(i_shape, 1)))
    model.add(tf.keras.layers.LSTM(units=16))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Reshape((1, 128))) # add a Reshape layer here
    model.add(tf.keras.layers.LSTM(units=8, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=4))
    model.add(tf.keras.layers.Dense(4))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(units=output))
    model.compile(optimizer='adam',
                loss='mean_squared_error', metrics=[direction_sensitive_loss, 'mae', mape_loss]) # loss=mape_loss
    model.summary()
    return model

def load_model(path: str):
    with custom_object_scope({'direction_sensitive_loss': direction_sensitive_loss, 'mape_loss': mape_loss}):
        return tf.keras.models.load_model(path)