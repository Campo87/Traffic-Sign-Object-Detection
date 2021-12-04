import glob
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models


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
        self.num_classes = len(self.labels)

        X = self.df.iloc[:, [0]]
        y = self.df.iloc[:, [1]]

        self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(X, y, test_size=0.2,
                                                                                random_state=1, stratify=y)


    def display_images(self):
        for idx, row in self.df.iterrows():
            cv2.imshow(winname=self.labels[row['Label_Encoded']], mat=row['Image'])
            cv2.waitKey(0)
    
    def model(self):
        self.model = tf.keras.Sequential([
            layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(180, 180, 3)),
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
        
        #self.X_train =  tf.convert_to_tensor(self.X_train)
        #self.X_validate =  tf.convert_to_tensor(self.X_validate)
        self.history = self.model.fit(x=self.X_train, y=self.y_train, epochs=5,
                                      validation_data=(self.X_validate, self.y_validate))



    def model_create_compile(self):
        pass


def main():
    sd = SignDetection()
    sd.model()
    del sd


if __name__ == '__main__': main()
