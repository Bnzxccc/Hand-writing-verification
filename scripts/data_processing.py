import tensorflow as tf
from tensorflow import keras
import os
import random

class MapFunction:
  def __init__(self, imageSize):
    # define image size
    self.imageSize = imageSize

  def decode_resize(self, filePath):
    # transform the image by path
    raw_image = tf.io.read_file(filePath)
    image = tf.image.decode_image(raw_image, channels=3)
    image = tf.image.resize(image, self.imageSize)
    image = tf.ensure_shape(image, (self.imageSize[0], self.imageSize[1], 3))

    return image

  def __call__(self, anchor, positive, negative):
    anchor = self.decode_resize(anchor)
    positive = self.decode_resize(positive)
    negative = self.decode_resize(negative)

    return anchor, positive, negative

class TripletGenerator:
  def __init__(self, datasetPath, n_samples):
    self.datasetPath = datasetPath
    self.validWriters = set() # classes with more than 1 samples
    self.articleCount = dict()
    self.n_samples = n_samples

    for filePath in os.listdir(self.datasetPath):
      writer = filePath.split('-')[0]

      if writer not in self.articleCount:
        self.articleCount[writer] = 1
      else:
        self.articleCount[writer] += 1

        if self.articleCount[writer] >= 2 and writer not in self.validWriters:
          self.validWriters.add(writer)

    self.articleByWriter = {writer: [] for writer in self.validWriters}
    # dictionary of class names as keys and list of image paths as values

    for filePath in os.listdir(self.datasetPath):
      writerName = filePath.split('-')[0]
      if writerName in self.validWriters:
        fullFilePath = os.path.join(self.datasetPath, filePath)
        self.articleByWriter[writerName].append(fullFilePath)


  def __call__(self):
    for _ in range(self.n_samples):
      tempValidWriters = list(self.validWriters)

      anchorClass = random.choice(tempValidWriters)
      tempValidWriters.remove(anchorClass)

      negativeClass = random.choice(tempValidWriters)

      anchorImage, positiveImage = random.sample(self.articleByWriter[anchorClass], 2)
      negativeImage = random.choice(self.articleByWriter[negativeClass])

      yield (anchorImage, positiveImage, negativeImage)
