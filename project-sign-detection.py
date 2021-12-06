import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
from tensorflow.python.ops.gen_math_ops import mod
from tensorflow.python.keras import Sequential
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

IMG_HEIGHT = 32
IMG_WIDTH = 32
CHANNELS = 3

class SignDetection():
    num_classes = 42
    def __init__(self):
        self.read_training_data()
    
    def read_training_data(self):
        data_dir = 'data/images'
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=32
        )
        
        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=32
        )
        self.class_names = self.train_ds.class_names
        #plt.figure(figsize=(10, 10))
        #for images, labels in self.train_ds.take(1):
        #    for i in range(9):
        #        ax = plt.subplot(3, 3, i + 1)
        #        plt.imshow(images[i].numpy().astype("uint8"))
        #        plt.title(class_names[labels[i]])
        #        plt.axis("off")
        #plt.waitforbuttonpress()
    
    def model(self):
        self.model = Sequential([
            layers.Rescaling(1./255),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
            layers.MaxPooling2D(),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes)
        ])

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.history_train = self.model.fit(self.train_ds, epochs=5)

        self.history_val = self.model.evaluate(self.val_ds)
        self.model.save("model/cnn_model.h5")
        self.model.save_weights("model/cnn_model_weight.h5")

    def load_model(self):
        self.model = tf.keras.models.load_model("model/cnn_model.h5")

        image = cv2.resize(cv2.imread('test.png'), (IMG_HEIGHT, IMG_WIDTH))
        image = np.expand_dims(image, axis=0)

        pred = self.model.predict(image)
        result = pred.argmax()
        print(pred)
        print(pred[0][result])
        print(self.class_names[result])


def main():
    sd = SignDetection()
    sd.model()
    #sd.load_model()
    del sd


if __name__ == '__main__': main()
