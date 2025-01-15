import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import resnet_v2

class SiameseModel(keras.Model):
  """
    A custom Siamese network model for learning embeddings and computing triplet loss.

  Args:
        imageSize: Tuple of integers (height, width). Specifies the input image dimensions.
        embedder: A Keras model that extracts embeddings from input images.
        margin: Float. The margin value for the triplet loss function.
        lossTracker: A Keras metric object used to track and report loss values during training.
  """
  def __init__(self, imageSize, embedder, margin, lossTracker):
    super().__init__()
    self.siameseNetwork = self.getSiameseModel(input_shape = imageSize + (3,), embedder = embedder)
    self.margin = margin
    self.lossTracker = lossTracker

  def getSiameseModel(self, input_shape, embedder):
    anchor_input = keras.Input(input_shape, name="anchor")
    positive_input = keras.Input(input_shape, name="positive")
    negative_input = keras.Input(input_shape, name="negative")

    anchor_embedding = embedder(anchor_input)
    positive_embedding = embedder(positive_input)
    negative_embedding = embedder(negative_input)

    siamese_network = keras.Model(inputs=[anchor_input, positive_input, negative_input],
                                  outputs=[anchor_embedding, positive_embedding, negative_embedding],
                                  name="Siamese_Network")

    return siamese_network

  def _compute_distance(self, inputs):
    anchor, positive, negative = inputs

    outputs = self.siameseNetwork([anchor, positive, negative])
    anchor_embedding = outputs[0]
    positive_embedding = outputs[1]
    negative_embedding = outputs[2]

    apDistance = tf.reduce_sum(tf.square(anchor_embedding - positive_embedding), axis = -1)
    anDistance = tf.reduce_sum(tf.square(anchor_embedding - negative_embedding), axis = -1)

    return (apDistance, anDistance)

  def _compute_loss(self, inputs):
    apDistance, anDistance = self._compute_distance(inputs)
    loss = tf.maximum(apDistance - anDistance + self.margin, 0)
    return tf.reduce_mean(loss)

  def train_step(self, inputs):
    with tf.GradientTape() as tape:
      loss = self._compute_loss(inputs)

    gradients = tape.gradient(
        loss,
        self.siameseNetwork.trainable_weights
    )

    self.optimizer.apply_gradients(
        zip(gradients, self.siameseNetwork.trainable_weights)
    )
    self.lossTracker.update_state(loss)
    return {'loss': self.lossTracker.result()}

  def test_step(self, inputs):
    loss = self._compute_loss(inputs)

    self.lossTracker.update_state(loss)
    return {'loss': self.lossTracker.result()}

  @property
  def metrics(self):
    return [self.lossTracker]

  def get_config(self):
    config = super(SiameseModel, self).get_config()
    config.update({"margin": self.margin})
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
