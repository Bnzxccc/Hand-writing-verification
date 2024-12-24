import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import resnet_v2

def EmbeddingModule(imageSize):
  base_model = resnet_v2.ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)
  for layer in base_model.layers:
    layer.trainable = False


  input = keras.Input(imageSize + (3,))
  x = resnet_v2.preprocess_input(input)

  extracted_features = base_model(x)

  x = layers.GlobalAveragePooling2D()(extracted_features)
  x = layers.Dense(512, activation='relu')(x)
  x = layers.Dropout(0.4)(x)
  x = layers.BatchNormalization()(x)
  x = layers.Dense(256, activation='relu')(x)
  x = layers.Dropout(0.4)(x)
  output = layers.Dense(128, activation = 'relu')(x)

  embedding = keras.Model(input, output, name = 'Embedding Module')

  return embedding


class SiameseModel(keras.Model):
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
                                  name="Siamese Network")

    return siamese_network

  def _compute_distance(self, inputs):
    anchor, positive, negative = inputs

    outputs = self.siameseNetwork([anchor, positive, negative])
    anchor_embedding = outputs[0]
    positive_embedding = outputs[1]
    negative_embedding = outputs[2]

    apDistance = tf.reduce_sum(tf.square(anchor_embedding - positive_embedding))
    anDistance = tf.reduce_sum(tf.square(anchor_embedding - negative_embedding))

    return (apDistance, anDistance)

  def _compute_loss(self, inputs):
    apDistance, anDistance = self._compute_distance(inputs)
    loss = tf.maximum(apDistance - anDistance + self.margin, 0)
    return loss

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
