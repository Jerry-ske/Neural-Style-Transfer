# Neural-Style-Transfer 
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from PIL import Image
import numpy as np

# Helper function to load and preprocess images
def load_image(image_path, image_size=(256, 256)):
    img = Image.open(image_path).resize(image_size)
    img = np.array(img) / 255.0
    img = img.astype(np.float32)
    return tf.constant(img)[tf.newaxis, ...]

# Load content and style images
content_image = load_image('content.jpg')
style_image = load_image('style.jpg')

# Load pre-trained style transfer model
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Apply style transfer
stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]

# Show images
def show_images(content, style, stylized):
    plt.figure(figsize=(12, 4))
    
    titles = ['Content Image', 'Style Image', 'Stylized Image']
    images = [content, style, stylized]
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i][0])
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

show_images(content_image, style_image, stylized_image)
