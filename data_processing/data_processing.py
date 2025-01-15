import tensorflow as tf
from tensorflow import keras
import os
import random
import tensorflow_io as tfio

class MapFunction:
  def __init__(self, imageSize):
    # define image size
    self.imageSize = imageSize

  def decode_resize(self, filePath):
     """
      Decodes and resizes the image from the file path.

      Args:
          filePath (str): Path to the image file.

      Returns:
          Tensor: Preprocessed image tensor.
      """
    raw_image = tf.io.read_file(filePath)
    image = tfio.experimental.image.decode_tiff(raw_image)
    image = image[:,:,:3]
    image = tf.image.resize(image, self.imageSize)
    image = tf.ensure_shape(image, (self.imageSize[0], self.imageSize[1], 3))

    return image

  def __call__(self, anchor, positive, negative):
    anchor = self.decode_resize(anchor)
    positive = self.decode_resize(positive)
    negative = self.decode_resize(negative)

    return anchor, positive, negative

class TripletGenerator:
  def __init__(self, datasetPath):
    self.datasetPath = datasetPath
    self.validWriters = set() # classes with more than 1 samples
    self.articleCount = dict()

    # Extract labels
    for filePath in os.listdir(self.datasetPath):
      writer = filePath.split('-')[0]

      if writer not in self.articleCount:
        self.articleCount[writer] = 1
      else:
        self.articleCount[writer] += 1

        if self.articleCount[writer] >= 2 and writer not in self.validWriters:
          self.validWriters.add(writer)

    # dictionary of class names as keys and list of image paths as values
    self.articleByWriter = {writer: [] for writer in self.validWriters}

    for filePath in os.listdir(self.datasetPath):
      writerName = filePath.split('-')[0]
      if writerName in self.validWriters:
        fullFilePath = os.path.join(self.datasetPath, filePath)
        self.articleByWriter[writerName].append(fullFilePath)


  def __call__(self):
    while True:
      tempValidWriters = list(self.validWriters)

      anchorClass = random.choice(tempValidWriters)
      tempValidWriters.remove(anchorClass)

      negativeClass = random.choice(tempValidWriters)

      anchorImage, positiveImage = random.sample(self.articleByWriter[anchorClass], 2)
      negativeImage = random.choice(self.articleByWriter[negativeClass])

      yield (anchorImage, positiveImage, negativeImage)
