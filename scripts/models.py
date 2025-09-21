from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import tensorflow as tf


def build_lstm_model(
    input_shape,
    lstm_units=[64],
    activation='tanh',
    dropout_rate=0.2,
    optimizer='adam',
    learning_rate=None,
    use_bidirectional=False
):
    """
    Builds and compiles an LSTM or Bidirectional LSTM model.

    Args:
        input_shape (tuple): Shape of input (timesteps, features)
        lstm_units (list of int): Number of units per LSTM layer
        activation (str): Activation function for LSTM layers
        dropout_rate (float): Dropout rate after each LSTM layer
        optimizer (str): 'adam', 'rmsprop', or 'sgd'
        learning_rate (float or None): Optional learning rate for optimizer
        use_bidirectional (bool): Whether to use Bidirectional LSTM layers

    Returns:
        Compiled Keras model
    """
    model = Sequential()

    for i, units in enumerate(lstm_units):
        return_sequences = i < len(lstm_units) - 1
        if use_bidirectional:
            lstm_layer = Bidirectional(
                LSTM(units, activation=activation, return_sequences=return_sequences),
                input_shape=input_shape if i == 0 else None
            )
        else:
            lstm_layer = LSTM(units, activation=activation, return_sequences=return_sequences,
                              input_shape=input_shape if i == 0 else None)

        model.add(lstm_layer)

        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    # Output layer for regression
    model.add(Dense(1))

    # Configure optimizer with custom learning rate
    if learning_rate:
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=learning_rate)
        else:
            raise ValueError("Unsupported optimizer")
    else:
        opt = optimizer  # fallback if optimizer already configured

    # Compile the model with RMSE metric
    model.compile(
        optimizer=opt,
        loss='mse',
        metrics=[lambda y, y_pred: tf.sqrt(tf.reduce_mean(tf.square(y - y_pred)))]  # RMSE
    )

    return model
