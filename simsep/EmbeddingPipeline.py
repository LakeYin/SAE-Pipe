from simsep import CosimilarityMatrix

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import numpy as np
import random

class EmbeddingPipeline:
    def __init__(self, model: Model, output_shape, loss=MeanSquaredError(), optimizer=Adam(), batch_size=None, epochs=10):
        self.model = model
        self.epochs = epochs
        self.output_shape = output_shape
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size

    def train(self, X: dict, M: CosimilarityMatrix):
        transformed = {k: np.random.normal(size=self.output_shape) for k in X.keys()}

        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            total_loss = 0
            
            if self.batch_size is None:
                samples = list(M)
                random.shuffle(samples)

                for label1, label2, score in samples:
                    transformed[label1], loss_value = self._train_step(X[label1], transformed[label2], score)
                    total_loss += loss_value
            
            print(f"Total loss: {total_loss}")

        return self

    def train(self, X: dict)
                
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

