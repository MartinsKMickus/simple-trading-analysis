import tensorflow as tf
import keras.backend as K


def mape_loss(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
    return 100. * K.mean(diff, axis=-1)
    

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
                    loss=mape_loss)#, metrics=['mae']) loss: mean_squared_error
    model.summary()
    return model