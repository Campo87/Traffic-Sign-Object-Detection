import os
import cv2
import glob

dirs = ['exclusionary', 'lane-usage', 'one-way', 'railroad',
        'school', 'speedlimit', 'stop-yield', 'warning']

for dir in dirs:
    images = [[cv2.imread(file), os.path.basename(file)] for file in glob.glob(f"original/{dir}/*.png")]
    for image in images:
        
        cv2.imwrite(f"{dir}/{image[1]}", cv2.resize(image[0], (64, 64)))
