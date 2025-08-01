import os
import glob
import tensorflow as tf
from tensorflow.keras import layers,models,datasets
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

model = load_model('unet_2_model.h5')

def custom_image_test(path):
    def load_and_preprocess_image(image_path, target_size):
        # Load the image
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    
    image_path = path  #'1_005_frame0007_leftImg8bit.jpg'
    target_size = (128, 128)
    input_image = load_and_preprocess_image(image_path, target_size)
    
    predicted_mask = model.predict(input_image)
    
    predicted_mask = tf.squeeze(tf.cast(predicted_mask > 0.5, tf.uint8))
    
    plt.figure(figsize=(4, 2))
    
    plt.subplot(1, 2, 1)
    plt.imshow(input_image.squeeze())
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")
    
    plt.tight_layout()
    #plt.show()
    
    print("Output:", predicted_mask.shape)
    
import glob
import time

for path in glob.glob('custom_test_images/*.jpg'):
    t1 = time.time()
    custom_image_test(path)
    t2 = time.time()
    print(int((t2-t1)*1000),"ms")