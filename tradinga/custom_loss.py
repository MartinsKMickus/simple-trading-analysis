import tensorflow as tf

def direction_loss(y_true, y_pred):
    """
    Loss is much higher if predicted value trend is wrong
    Args:
        symbol (str): Symbol name.
    """
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
        1000 * loss,  # Increase the loss for the special occasion
        loss
    )
    
    loss = tf.where(
        tf.logical_and(tf.less(y_true, prev_true), tf.greater(y_pred, prev_true)),
        1000 * loss,  # Increase the loss for the special occasion
        loss
    )
    
    return loss

class IntervalAccuracy(tf.keras.metrics.Accuracy):
    def __init__(self, interval=0.03, name='interval_accuracy', dtype=None):
        super(IntervalAccuracy, self).__init__(name=name, dtype=dtype)
        self.interval = interval

    def update_state(self, y_true, y_pred, sample_weight=None):

        # Compute the absolute difference between y_pred and y_true
        abs_diff = tf.abs(y_pred - y_true)

        # Determine if the absolute difference is within the specified interval
        within_interval = tf.less_equal(abs_diff, self.interval)

        # Compute the accuracy based on whether the predictions are within the interval
        acc = tf.cast(within_interval, self._dtype)

        super(IntervalAccuracy, self).update_state(y_true, acc, sample_weight)