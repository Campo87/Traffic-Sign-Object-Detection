import glob
import os
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from pandas.core.accessor import PandasDelegate
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator


class SignDetection():
    df = pd.DataFrame(columns=['Image', 'Label'])
    labels = list()
    num_classes = 0
    def __init__(self):
        self.read_training_data()
    
    def read_training_data(self):
        images = [[cv2.imread(file), os.path.basename(file)] for file in glob.glob(f"data/images/*.png")]
        for image in images:
            self.df = self.df.append({'Image': image[0],
                           'Label': image[1][:-6]}, ignore_index=True)

        le = preprocessing.LabelEncoder()
        le.fit(self.df['Label'])
        self.df['Label_Encoded'] = le.transform(self.df['Label'])
        self.labels = le.inverse_transform(self.df['Label_Encoded'])
        self.labels = list(dict.fromkeys(self.labels))
        print(self.labels)
        self.num_classes = len(self.labels)

        print(self.labels)

        X = self.df.iloc[:, [0]]
        y = self.df.iloc[:, [1]]


    def display_images(self):
        for idx, row in self.df.iterrows():
            cv2.imshow(winname=self.labels[row['Label_Encoded']], mat=row['Image'])
            cv2.waitKey(0)
    
    def model(self):
        self.resize_rescale = tf.keras.Sequential([
            # Resize and rescale
            layers.Resizing(180, 180),
            layers.Rescaling(1./255)
        ])


        self.data_augmentation = tf.keras.Sequential([
            # Alter image
            layers.RandomContrast(factor=0.1),
            layers.RandomRotation(factor=0.05),
            layers.Resizing(height=180, width=180)
        ])

        self.model = tf.keras.Sequential([
            self.resize_rescale,
            self.data_augmentation,
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),

            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes)
        ])

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        
        self.history = self.model.fit(x=self.X_train, y=self.y_train, epochs=5,
                                      validation_data=(self.X_val, self.y_val))



    def model_create_compile(self):
        pass


def main():
    sd = SignDetection()
    sd.display_images()
    del sd


if __name__ == '__main__': main()
