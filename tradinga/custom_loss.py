import tensorflow as tf

def direction_loss(y_true, y_pred):
    # Calculate the previous values of y_true
    prev_true = tf.roll(y_true, shift=1, axis=0)
    
    # Calculate the loss based on different scenarios
    loss = tf.where(
        tf.logical_and(tf.greater(y_true, prev_true), tf.greater(y_pred, prev_true)),
        tf.abs(y_pred - y_true),
        tf.abs(y_pred - y_true)  # Use a different loss calculation for other cases
    )
    
    # Apply additional penalties for special occasions
    loss = tf.where(
        tf.logical_and(tf.greater(y_true, prev_true), tf.less(y_pred, prev_true)),
        10 * loss,  # Increase the loss for the special occasion
        loss
    )
    
    loss = tf.where(
        tf.logical_and(tf.less(y_true, prev_true), tf.greater(y_pred, prev_true)),
        10 * loss,  # Increase the loss for the special occasion
        loss
    )
    
    return loss