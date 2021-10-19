from saepipe import DifferenceMatrix

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar
import numpy as np
import random

class EmbeddingPipeline:

    def __init__(self, model: Model, output_shape, loss=MeanSquaredError(), optimizer=Adam(), ):
        self.model = model
        self.output_shape = output_shape
        self.loss = loss
        self.optimizer = optimizer

    def train(self, diff_func, X, y=None, init='random', scale=None, batch_size=None, epochs=10, verbose=1):
        if (y is not None) and (isinstance(y, dict) is not isinstance(X, dict)):
            raise TypeError("Cannot match labels to data.")

        if (y is not None) and (not isinstance(y, dict) and not isinstance(X, dict)):
            y = {i: y for i, y in enumerate(y)}

        if not isinstance(X, dict):
            X = {i: x for i, x in enumerate(X)}

        M = DifferenceMatrix(diff_func, X, labels=y, scale=scale)

        if init == 'random':
            transformed = {k: np.random.normal(size=self.output_shape) for k in X.keys()}
        elif init == 'zero':
            transformed = {k: np.zeros(self.output_shape) for k in X.keys()}
        else:
            raise ValueError("init arg must be 'random' or 'zero'.")

        for epoch in range(epochs):
            if verbose > 0:
                print(f"Epoch {epoch + 1}/{epochs}")

            total_loss = 0
            
            if batch_size is None:
                samples = list(M)
                random.shuffle(samples)
            else:
                pass

            bar = Progbar(len(samples), verbose=verbose)

            for label1, label2, score in samples:
                transformed[label1], loss_value = self._train_step(X[label1], transformed[label2], score)
                total_loss += loss_value

                bar.add(1)
            
            if verbose > 0:
                print(f"Total loss: {total_loss}")

        return self
                
    def predict(self, x):
        return self.model.predict(x)

    def _train_step(self, x, y, score):
        if not tf.is_tensor(x) and x.ndim < 2:
            x = np.reshape(x, (1, x.size))

        with tf.GradientTape() as tape:
            output = self.model(x, training=True)

            loss_value = self.loss([score], [tf.norm(output - y)])
            loss_value += sum(self.model.losses)

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        return self.model(x), loss_value
