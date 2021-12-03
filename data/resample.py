import glob
import os

import cv2
import numpy as np
import tensorflow as tf
from keras import layers

# Update this if new sign categorgies are added
dirs = ['exclusionary', 'lane-usage', 'one-way', 'railroad',
        'school', 'speedlimit', 'stop-yield', 'warning']

# Holds the image and image path
images = []

# Search each categories for png images
for dir in dirs:
        images += [[cv2.imread(file), os.path.basename(file)] for file in glob.glob(f"original/{dir}/*.png")]

# Setup data augementation model
data_augmentation = tf.keras.Sequential([
        layers.Resizing(64, 64),
        layers.RandomContrast(factor=0.5),
        layers.RandomRotation(factor=0.02),
        layers.RandomHeight(factor=0.2),
        layers.RandomWidth(factor=0.2)
])

# Augment the data 'n save it
for image in images:
        # Make 10 augmented copies of each type of sign
        for idx in range(0, 10):
                result = np.array(data_augmentation(image[0]))
                cv2.imwrite(f"images/{image[1][:-4]}_{idx}.png", result)
