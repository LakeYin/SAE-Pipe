from saepipe import DifferenceMatrix

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import Progbar
import numpy as np
import random

class EmbeddingPipeline:

    def __init__(self, model: Model, output_shape):
        self._model = model
        self._output_shape = output_shape

    def train(self, diff_func, X, y=None, init='zero', scale=None, batch_size=None, epochs=10, verbose=1):
        M = DifferenceMatrix(diff_func, X, labels=y, scale=scale)

        if init == 'random':
            transformed = np.random.normal(size=(len(X), self._output_shape))
        elif init == 'zero':
            transformed = np.zeros((len(X), self._output_shape))
        else:
            raise ValueError("init arg must be 'random' or 'zero'.")

        if batch_size is None:
            diffs = [(i, j, tf.constant([d])) for i, j, d in M if i != j]
        else:
            pass

        for epoch in range(epochs):
            if verbose > 0:
                print(f"Epoch {epoch + 1}/{epochs}")

            random.shuffle(diffs)

            bar = Progbar(len(diffs), verbose=verbose)

            total_loss = 0
            for index1, index2, diff in diffs:
                transformed[index1], loss_value = self._train_step(X[index1], transformed[index2], diff)
                total_loss += loss_value

                bar.add(1)
            
            if verbose > 0:
                print(f"Total loss: {total_loss}")

        return self
                
    def embed(self, x):
        return self._model.predict(x)

    def _train_step(self, a, b_embed, diff):
        if isinstance(a, np.ndarray) and a.ndim < 2:
            a = np.reshape(a, (1, a.size))

        with tf.GradientTape() as tape:
            a_embed = self._model(a, training=True)

            loss = self._model.compiled_loss(diff, tf.reshape(tf.norm(a_embed - b_embed), 1), regularization_losses=self._model.losses)

        trainable_vars = self._model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self._model.optimizer.apply_gradients(zip(grads, trainable_vars))

        return self._model(a), loss
