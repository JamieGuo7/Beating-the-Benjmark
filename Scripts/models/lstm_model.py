from .base_predictor import BasePredictor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam


def directional_loss(y_true, y_pred):
    """ Custom loss function which penalises incorrect direction """
    huber = tf.keras.losses.Huber()(y_true, y_pred)
    sign_penalty = tf.reduce_mean(
        tf.abs(tf.sign(y_true) - tf.sign(y_pred))
    )
    return huber + 0.5 * sign_penalty

class LSTMPredictor:
    """ LSTM Model for time series prediction """

    def __init__(self, input_shape, learning_rate=0.005):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Input(shape=self.input_shape),
            LSTM(units=32, return_sequences=True),
            Dropout(0.2),

            LSTM(units=16, return_sequences=False),
            Dropout(0.2),

            Dense(units=16),
            LeakyReLU(0.1),

            Dense(units=1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=directional_loss,
            metrics=['mae']
        )

        return model

    def train(self, X_train, y_train, X_val, y_val,
              epochs=100, batch_size=32, callbacks=None):
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0 # Tells the model not to print updates while training
        )
        return history

    def predict(self, X, verbose=0):
        return self.model.predict(X, verbose=verbose).flatten()

    def save(self, filepath):
        self.model.save(filepath)