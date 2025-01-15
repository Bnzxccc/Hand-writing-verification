import tensorflow as tf
from tensorflow import keras
from keras import layers

def _pairwise_distances(embedding_batch):
    """
    Compute the matrix containing pairwise distances of all embeddings in a batch

    Args:
        embedding_batch (str): tensor of shape (batch_size, embed_dim)

    Returns:
        tensor of shape (batch_size, batch_size)
    """

    # compute dot product of all pairs of embeddings, shaped (batch_size, batch_size)
    dot_prod = tf.matmul(embedding_batch, tf.transpose(embedding_batch))

    # extract the squared l2_norm shaped (batch_size,)
    squared_norm = tf.diag_part(dot_prod)

    # compute pairwise distance of all pairs of squared norms of embeddings
    pairwise = tf.expand_dims(squared_norm, 1) - 2.0 * dot_prod + tf.expand_dims(squared_norm, 0)

    # put all negative distances to 0 (computational errors)
    return tf.maximum(pairwise, 0)

def _get_anchor_positive_mask(labels):
    """
    Returns mask where mask[a,p] is true if a and p are of the same label and distinct

    Args:
        labels (str): tensor of shape (batch_size, )

    Returns:
        tensor of shape (batch_size, batch_size)
    """

    





class HardTripletSiameseModel(keras.Model)