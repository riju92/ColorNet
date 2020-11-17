import argparse
import random
import matplotlib.pyplot as plt
import cv2
import os
import glob
from torchvision import transforms
from sklearn.model_selection import train_test_split


#image augmentation script
img_dir = "./face_images/*.jpg"
files = glob.glob(img_dir)
random.shuffle(files)

#split the dataset into train and test set
train_data, test_data = train_test_split(files, train_size = int(len(files)*90/100), test_size = int(len(files)*10/100))
i = 0
for f in train_data:
    img = cv2.imread(f)
    cv2.imwrite("train_data/train_image%05i.jpg" %i, img)
    i+=1

i = 0
for f in test_data:
    img = cv2.imread(f)
    cv2.imwrite("test_data/test_image%05i.jpg" %i, img)
    i+=1

i = 0
restart = True
train_dir = "./train_data/*.jpg"
train_files = glob.glob(train_dir)
while restart:
    random.shuffle(train_files)
    for f1 in train_files:
        img = cv2.imread(f1)
        x,y = img.shape[0], img.shape[1]
        # 1 for horizontal flip
        img = cv2.flip(img, flipCode=1)
        #random cropping 
        crop_x = random.randint(1, x)
        crop_y = random.randint(1, y)
        img = img[0:0 + crop_y, 0:0 + crop_x]
        img = cv2.resize(img, (x,y))
        # scaling RGB
        scaler = random.uniform(0.6,1.0)
        b, g, r = cv2.split(img)
        b_scaled = cv2.multiply(b, scaler)
        g_scaled = cv2.multiply(g, scaler)
        r_scaled = cv2.multiply(r, scaler)
        rescaled_img = cv2.merge((b_scaled, g_scaled, r_scaled))
        cv2.imwrite("aug_images/image%05i.jpg" %i, rescaled_img)
        i+=1
        if i == len(train_files) * 10:
            restart = False
            break
        else:
            continue    
