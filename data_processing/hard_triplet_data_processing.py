import tensorflow as tf
from tensorflow import keras
import random
import os
import tensorflow_io as tfio

class HardTripletDataGenerator:
    def __init__(self, datasetPath):
        self.datasetPath = datasetPath
        self.validWriters = set() # classes with more than 1 samples
        self.articleCount = dict()

        # Extract the labels
        for filePath in os.listdir(self.datasetPath):
            writer = filePath.split('-')[0]

            if writer not in self.articleCount:
                self.articleCount[writer] = 1
            else:
                self.articleCount[writer] += 1

                if self.articleCount[writer] >= 2 and writer not in self.validWriters:
                self.validWriters.add(writer)

        # dictionary of image paths as keys and labels as values
        self.articleToWriter = {}

        for filePath in os.listdir(self.datasetPath):
            writerName = filePath.split('-')[0]
            if writerName in self.validWriters:
                fullFilePath = os.path.join(self.datasetPath, filePath)
                self.articleToWriter[fullFilePath] = writerName

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

    def __call__(self):
        """
        Yields random image-label pairs for training.

        Returns:
            Tuple: (image, label)
        """
        while True:
            # Randomly select an image path
            randomImagePath = random.choice(self.articleToWriter)
            label = self.articleToWriter[randomImagePath]  # Get corresponding label

            image = self.decode_resize(randomImagePath)

            yield image, label