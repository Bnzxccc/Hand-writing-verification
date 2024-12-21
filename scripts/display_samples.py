import matplotlib.pyplot as plt
import tensorflow as tf
import random

def display_triplets(dataset, num_images=1):
    # Get one batch from the dataset
    iterator = iter(dataset)
    anchor_batch, positive_batch, negative_batch = next(iterator)

    # Limit the number of triplets to display
    num_images = min(num_images, anchor_batch.shape[0], 25)

    # Randomly select indices from the batch
    random_indices = random.sample(range(anchor_batch.shape[0]), num_images)

    # Set up the figure: num_images rows, 3 columns (A, P, N)
    plt.figure(figsize=(15, 5 * num_images), dpi=200)

    for i, idx in enumerate(random_indices):
        print("Anchor shape:", anchor_batch[idx].shape)
        # Anchor
        ax = plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(anchor_batch[idx].numpy().astype("uint8"))
        plt.title("Anchor")
        plt.axis("off")

        # Positive
        ax = plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(positive_batch[idx].numpy().astype("uint8"))
        plt.title("Positive")
        plt.axis("off")

        # Negative
        ax = plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(negative_batch[idx].numpy().astype("uint8"))
        plt.title("Negative")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
