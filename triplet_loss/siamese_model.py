import tensorflow as tf
from tensorflow import keras
from keras import layers

class SiameseModel(keras.Model):
    """
    A custom Siamese network model for learning embeddings and computing triplet loss.

    Args:
        imageSize: Tuple (height, width) specifying input image dimensions.
        embedder: A Keras model to extract embeddings from input images.
        margin: Float, margin for triplet loss.
        lossTracker: A Keras metric object to track and report loss during training.
    """
    def __init__(self, imageSize, embedder, margin, lossTracker):
        super().__init__()
        # Define the Siamese network with an input shape and embedder model
        self.siameseNetwork = self.getSiameseModel(input_shape=imageSize + (3,), embedder=embedder)
        self.margin = margin
        self.lossTracker = lossTracker

    def getSiameseModel(self, input_shape, embedder):
        """
        Builds the Siamese network model, taking three inputs (anchor, positive, negative)
        and computing their embeddings using the provided embedder.
        
        Args:
            input_shape: Tuple representing the shape of input images.
            embedder: The model used to extract embeddings.

        Returns:
            A Keras Model with three inputs and three embeddings as outputs.
        """
        # Define input tensors for anchor, positive, and negative samples
        anchor_input = keras.Input(input_shape, name="anchor")
        positive_input = keras.Input(input_shape, name="positive")
        negative_input = keras.Input(input_shape, name="negative")

        # Extract embeddings for each input
        anchor_embedding = embedder(anchor_input)
        positive_embedding = embedder(positive_input)
        negative_embedding = embedder(negative_input)

        # Build and return the Siamese network model
        siamese_network = keras.Model(
            inputs=[anchor_input, positive_input, negative_input],
            outputs=[anchor_embedding, positive_embedding, negative_embedding],
            name="Siamese_Network"
        )
        return siamese_network

    def _compute_distance(self, inputs):
        """
        Computes the squared distances between anchor-positive and anchor-negative embeddings.
        
        Args:
            inputs: Tuple of tensors (anchor, positive, negative).
        
        Returns:
            Tuple of tensors (apDistance, anDistance), where:
              - apDistance: Distance between anchor and positive embeddings.
              - anDistance: Distance between anchor and negative embeddings.
        """
        anchor, positive, negative = inputs

        # Get embeddings from the Siamese network
        outputs = self.siameseNetwork([anchor, positive, negative])
        anchor_embedding = outputs[0]
        positive_embedding = outputs[1]
        negative_embedding = outputs[2]

        # Compute squared distances
        apDistance = tf.reduce_sum(tf.square(anchor_embedding - positive_embedding), axis=-1)
        anDistance = tf.reduce_sum(tf.square(anchor_embedding - negative_embedding), axis=-1)

        return (apDistance, anDistance)

    def _compute_loss(self, inputs):
        """
        Computes the triplet loss using the computed distances.
        
        Args:
            inputs: Tuple of tensors (anchor, positive, negative).
        
        Returns:
            A scalar tensor representing the mean triplet loss.
        """
        apDistance, anDistance = self._compute_distance(inputs)
        # Triplet loss with a margin
        loss = tf.maximum(apDistance - anDistance + self.margin, 0)
        return tf.reduce_mean(loss)

    def train_step(self, inputs):
        """
        Performs one training step: computes the loss, gradients, and applies them.
        
        Args:
            inputs: Tuple of tensors (anchor, positive, negative).
        
        Returns:
            A dictionary containing the current loss value.
        """
        with tf.GradientTape() as tape:
            loss = self._compute_loss(inputs)

        # Compute gradients and apply them
        gradients = tape.gradient(loss, self.siameseNetwork.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siameseNetwork.trainable_weights))

        # Update the loss tracker
        self.lossTracker.update_state(loss)
        return {'loss': self.lossTracker.result()}

    def test_step(self, inputs):
        """
        Performs one testing step: computes and tracks the loss.
        
        Args:
            inputs: Tuple of tensors (anchor, positive, negative).
        
        Returns:
            A dictionary containing the current loss value.
        """
        loss = self._compute_loss(inputs)
        self.lossTracker.update_state(loss)
        return {'loss': self.lossTracker.result()}

    @property
    def metrics(self):
        """
        Defines the metrics to track during training/testing.
        """
        return [self.lossTracker]

    def get_config(self):
        """
        Saves the model configuration, including the margin value.
        """
        config = super(SiameseModel, self).get_config()
        config.update({"margin": self.margin})
        return config

    @classmethod
    def from_config(cls, config):
        """
        Restores the model from its configuration.
        """
        return cls(**config)
