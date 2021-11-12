from saepipe import DifferenceMatrix

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.utils import Progbar
import numpy as np
import random

class EmbeddingPipeline(Sequential):
    
    def fit(self, diff_func, X, y=None, init='zero', scale=None, batch_size=None, epochs=10, verbose=1, shuffle=True):
        input_size = np.shape(X[0])
        self.build((1, input_size[0]) if len(input_size) < 2 else input_size)

        if len(self.layers[-1].output_shape) != 2:
            raise ValueError("Output layer must have a 2D output.")

        output_shape = self.layers[-1].output_shape[1]

        M = DifferenceMatrix(diff_func, X, labels=y, scale=scale)

        if init == 'random':
            transformed = np.random.normal(size=(len(X), output_shape)).astype(np.float32)
        elif init == 'zero':
            transformed = np.zeros((len(X), output_shape), dtype=np.float32)
        else:
            raise ValueError("init arg must be 'random' or 'zero'.")

        if batch_size is None:
            diffs = [(i, j, tf.constant([d], dtype=tf.float32)) for i, j, d in M if i != j]
        else:
            pass

        for epoch in range(epochs):
            if verbose > 0:
                print(f"Epoch {epoch + 1}/{epochs}")

            if shuffle:
                random.shuffle(diffs)
            
            bar = Progbar(len(diffs), verbose=verbose)

            total_loss = 0
            for index1, index2, diff in diffs:
                x = tf.constant(X[index1], dtype=tf.float32)
                shape = tf.shape(x)
                if len(shape) < 2:
                    x = tf.reshape(x, (1, shape[0]))

                transformed[index1], loss = self.train_step(x, transformed[index2], diff)
                total_loss += loss

                bar.add(1)
            
            if verbose > 0:
                print(f"Total loss: {total_loss}")

        return self.history

    @tf.function
    def train_step(self, a, b_embed, diff):
        with tf.GradientTape() as tape:
            embed_diff = tf.reshape(tf.norm(self(a, training=True) - b_embed), (1,))

            loss = self.compiled_loss(diff, embed_diff, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        self.compiled_metrics.update_state(diff, embed_diff)

        return self(a), loss
