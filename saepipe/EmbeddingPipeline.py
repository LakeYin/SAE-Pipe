from saepipe import DifferenceMatrix

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar
from tensorflow.keras.models import clone_model
import numpy as np
import random

class EmbeddingPipeline:

    def __init__(self, model: Model, output_shape, loss=MeanSquaredError(), optimizer=Adam()):
        self._model = clone_model(model)
        self._output_shape = output_shape
        self._loss = loss
        self._optimizer = optimizer

    def train(self, diff_func, X, y=None, init='random', scale=None, batch_size=None, epochs=10, verbose=1):
        M = DifferenceMatrix(diff_func, X, labels=y, scale=scale)

        if init == 'random':
            transformed = np.random.normal(size=(len(X), self._output_shape))
        elif init == 'zero':
            transformed = np.zeros((len(X), self._output_shape))
        else:
            raise ValueError("init arg must be 'random' or 'zero'.")

        if batch_size is None:
            scores = [(i, j, s) for i, j, s in M if i != j]
        else:
            pass

        for epoch in range(epochs):
            if verbose > 0:
                print(f"Epoch {epoch + 1}/{epochs}")

            random.shuffle(scores)

            bar = Progbar(len(scores), verbose=verbose)

            total_loss = 0
            for index1, index2, score in scores:
                transformed[index1], loss_value = self._train_step(X[index1], transformed[index2], score)
                total_loss += loss_value

                bar.add(1)
            
            if verbose > 0:
                print(f"Total loss: {total_loss}")

        return self
                
    def embed(self, x):
        return self._model.predict(x)

    def get_model(self):
        return clone_model(self._model)

    def _train_step(self, a, b, score):
        if isinstance(a, np.ndarray) and a.ndim < 2:
            a = np.reshape(a, (1, a.size))

        with tf.GradientTape() as tape:
            output = self._model(a, training=True)

            loss_value = self._loss([score], [tf.norm(output - b)])
            loss_value += sum(self._model.losses)

        grads = tape.gradient(loss_value, self._model.trainable_weights)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_weights))

        return self._model(a), loss_value
