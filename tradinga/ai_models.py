import tensorflow as tf


def model_v1(i_shape, output = 1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=64,
                                    return_sequences=True,
                                    input_shape=(i_shape, 1)))
    model.add(tf.keras.layers.LSTM(units=64))
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(units=output))
    model.compile(optimizer='adam',
                    loss='mean_squared_error')
    return model
    
def model_v2(i_shape, output = 1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=128, input_shape=(i_shape, 1)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=output))
    model.compile(optimizer='adam',
                    loss='mean_squared_error')
    return model


def model_v3(i_shape, output = 1):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=64, return_sequences=True, input_shape=(i_shape, 1)))
    model.add(tf.keras.layers.LSTM(units=16, return_sequences=True))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.LSTM(units=8))
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Dense(4))
    # Process and divide information (main start layer?)
    # Throughout these proceses try to drop some of the values
    # Next trending gathering?
    # Final decision
    #model.add(tf.keras.layers.LSTM(units=64))
    #model.add(tf.keras.layers.Dense(dense_after))
    #model.add(tf.keras.layers.Dropout(0.7))
    #model.add(tf.keras.layers.Reshape((dense_after, 1)))
    #model.add(tf.keras.layers.LSTM(units=10))
    #model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=output))
    model.compile(optimizer='adam',
                    loss='mean_squared_error', metrics=['mae'])
    model.summary()
    return model