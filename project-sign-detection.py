import os
import cv2
import glob
import pandas as pd

class SignDetection():
    training_dirs = ['exclusionary', 'lane-usage', 'one-way', 'railroad',
                     'school', 'speedlimit', 'stop-yield', 'warning']
    df = pd.DataFrame(columns=['Image', 'Label'])
    def __init__(self):
        self.read_training_data()
    
    def read_training_data(self):
        for dir in self.training_dirs:
            images = [[cv2.imread(file), os.path.basename(file)] for file in glob.glob(f"data/training/{dir}/*.png")]
            for image in images:
                self.df = self.df.append({'Image': image[0],
                               'Label': image[1][:-4]}, ignore_index=True)

    def display_images(self):
        for idx, row in self.df.iterrows():
            cv2.imshow(winname=row['Label'], mat=row['Image'])
            cv2.waitKey()


def main():
    sd = SignDetection()
    sd.display_images()
    del sd


if __name__ == '__main__': main()
