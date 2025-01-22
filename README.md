# -Training-Autoencoders-with-Keras-to-Interpolate-between-Handwritten-Digits-on-CoreML
To train an autoencoder in Python with Keras and then convert the trained model to CoreML for interpolation between handwritten digits (such as from the MNIST dataset), you'll need to break it down into a few distinct steps. Below is a complete guide with the code for each step:
Step-by-Step Approach:

    Install Required Libraries:
        First, make sure you have the necessary libraries installed for TensorFlow/Keras, CoreML, and other dependencies.
        Install them using pip if necessary:

        pip install tensorflow coremltools matplotlib numpy

    Build and Train the Autoencoder:
        The autoencoder consists of an encoder (to compress the input) and a decoder (to reconstruct the input).
        We'll use the MNIST dataset (which consists of handwritten digits) for training.

    Train the Autoencoder:
        We'll train the autoencoder on the MNIST dataset.

    Interpolate between Handwritten Digits:
        After training, we'll generate interpolations between two digits by performing linear interpolation in the latent space of the autoencoder.

    Convert the Model to CoreML:
        Finally, we'll convert the trained Keras model into a CoreML model, which can be used in iOS apps.

Complete Python Code:

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import coremltools
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize the data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Reshape the data to match the input shape of the network
x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

# Define the Autoencoder model
input_img = layers.Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)

# Latent space (bottleneck)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)

# Output layer
decoded = layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

# Compile the autoencoder
autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
autoencoder.fit(x_train, x_train, epochs=20, batch_size=128, validation_data=(x_test, x_test))

# Visualize a few original and reconstructed images
decoded_imgs = autoencoder.predict(x_test)

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

    # Display reconstructed images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

# Perform interpolation between two digits (latent space interpolation)
def interpolate(model, img1, img2, steps=10):
    # Get the encoder part of the autoencoder
    encoder = models.Model(inputs=model.input, outputs=model.layers[6].output)  # Output of encoder
    # Get the decoder part of the autoencoder
    latent1 = encoder.predict(np.expand_dims(img1, axis=0))
    latent2 = encoder.predict(np.expand_dims(img2, axis=0))
    
    # Linearly interpolate between the latent vectors
    interpolated_latents = []
    for alpha in np.linspace(0, 1, steps):
        interpolated_latent = latent1 * (1 - alpha) + latent2 * alpha
        decoded_image = model.decoder.predict(interpolated_latent)
        interpolated_latents.append(decoded_image)
    
    return np.array(interpolated_latents)

# Interpolation between two random digits
img1 = x_test[0]  # Take the first test image
img2 = x_test[1]  # Take the second test image
interpolated_images = interpolate(autoencoder, img1, img2, steps=10)

# Display the interpolated images
plt.figure(figsize=(20, 4))
for i in range(10):
    ax = plt.subplot(1, 10, i + 1)
    plt.imshow(interpolated_images[i].reshape(28, 28), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()

# Convert the trained Keras model to CoreML format
coreml_model = coremltools.convert(autoencoder, inputs=[coremltools.models.datatypes.Array(1, 28, 28, 1)])
coreml_model.save("autoencoder.mlmodel")

Explanation of the Code:

    Data Preprocessing:
        The MNIST dataset is loaded and normalized to be in the range [0, 1].
        The dataset is reshaped to fit the input shape required by the autoencoder (28x28x1 for grayscale images).

    Autoencoder Architecture:
        The autoencoder model is composed of a convolutional encoder and a convolutional decoder.
        The encoder compresses the image to a lower-dimensional latent space, and the decoder reconstructs the image from the latent representation.

    Training:
        The model is trained for 20 epochs using the binary cross-entropy loss and the Adam optimizer.

    Image Interpolation:
        After training, the model is used to generate interpolations between two test images. The interpolation is done by linearly interpolating between the two latent representations of the images and decoding them back into images.

    CoreML Conversion:
        Once the model is trained, it is converted into CoreML format using coremltools for use in iOS applications. The model is saved as autoencoder.mlmodel.

Next Steps:

    You can use the autoencoder.mlmodel in an iOS app to deploy the model and perform inference on devices.
    For interpolation, you could build an interactive app that allows users to pick two digits and see how the model interpolates between them.
