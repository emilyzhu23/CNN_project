import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import os
import random

def getImage():
    """
    Function: Gets image path from user and outputs original image
    Arguments: None
    Return Vals: imageo, image object for user-inputted, original image
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    args = vars(ap.parse_args())
    imageo = cv2.imread(args["image"])
    cv2.imshow('original', imageo)
    return imageo

def rotate_image(img, angle):
    rows = img.shape[0]
    cols = img.shape[1]

    img_center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(img_center, angle, 1)

    rotated_image = cv2.warpAffine(img, M, (cols, rows), borderMode = cv2.BORDER_REFLECT)
    return rotated_image

def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode = cv2.BORDER_REFLECT)
    return shifted


arr = os.listdir('flowerclasseddataset/') #len should be 102
print(len(arr))
for i in range(46, 103): # 1, 103
    dir = "flowerclasseddataset/" + str(i)
    arr = os.listdir(dir)
    numImgToAugment = 166 - len(arr)
    numOfTrans = numImgToAugment // len(arr) + 1
    j = 0
    while numImgToAugment > 0:
        print(numImgToAugment)
        print("folder i")
        print(i)
        imgo = arr[j % len(arr)]
        if imgo.endswith(".jpg"):
            print(imgo)
            img = cv2.imread("flowerclasseddataset/" + str(i) +"/" + str(imgo))
            cv2.imshow('image', img)
            if (j // len(arr) + 1) % 2 == 0:
                rIMG = rotate_image(img, random.randint(0, 360))
            else:
                rIMG = translate(img, random.randint(-100, 100), random.randint(-100, 100))
            filename = "flowerclasseddataset/" + str(i) + "/augment" + str(j // len(arr)) + "_" + str(imgo)
            cv2.imwrite(filename, rIMG)
            numImgToAugment -= 1
        else:
            pass
        j += 1
    
    imgNum = 0
    for j in range (1, numOfTrans + 1):
        imgNum += 1