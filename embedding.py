import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.applications import resnet_v2

def EmbeddingModule(imageSize):
    """
    Create an embedding module for feature extraction using ResNet50V2 as the base.

    This module extracts embeddings from input images by first utilizing a pre-trained ResNet50V2 
    network to extract features, followed by additional dense layers to generate compact embeddings. 
    The base ResNet50V2 is frozen to retain pre-trained weights, ensuring transfer learning is applied.

    Args:
        imageSize: Tuple of integers (height, width). Specifies the input image dimensions.

    Returns:
        embedding: A Keras Model that takes an image as input and outputs embedding of shape (batch_size,128).
    """

    base_model = resnet_v2.ResNet50V2(weights='imagenet', include_top=False, input_shape=imageSize + (3,))
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

    embedding = keras.Model(input, output, name = 'Embedding_Module')

    return embedding

