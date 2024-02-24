import numpy as np
import csv
import glob
import os
import cv2
import time
import seaborn as sns
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import random
from cut import cut
from compute import compute

def drawFeature(inputPath_defective, inputPath_qualified):
    qualifiedFiles = glob.glob(os.path.join(inputPath_qualified, '*.jpg'))
    qualifiedBeanImages = []
    qualifiedCount = 0
    for file in qualifiedFiles:
        image = cv2.imread(file)
        beans = cut(image)
        qualifiedBeansContours = [bean for bean in beans if cv2.contourArea(bean) > 163]
        for contour in qualifiedBeansContours:
            x, y, w, h = cv2.boundingRect(contour)
            qualifiedBeanImages.append(image[y:y+h, x:x+w])
        qualifiedCount = qualifiedCount + len(qualifiedBeansContours)

    defectiveFiles = glob.glob(os.path.join(inputPath_defective, '*.jpg'))
    defectiveBeanImages = []
    defectiveCount = 0
    for file in defectiveFiles:
        image = cv2.imread(file)
        beans = cut(image)
        defectiveBeansContours = [bean for bean in beans if cv2.contourArea(bean) > 163]
        for contour in defectiveBeansContours:
            x, y, w, h = cv2.boundingRect(contour)
            defectiveBeanImages.append(image[y:y+h, x:x+w])
        defectiveCount = defectiveCount + len(defectiveBeansContours)

    m = min(defectiveCount, qualifiedCount)
    sample_num = min(30, int(m*0.3))
    random_numbers = random.sample(range(0, m), sample_num)

    sns.set(rc={'figure.figsize':(7.5,5)})
    # plot qualified feature
    for i in random_numbers:
        singleBean = qualifiedBeanImages[i]
        R = singleBean[:, :, 2].ravel()
        G = singleBean[:, :, 1].ravel()
        B = singleBean[:, :, 0].ravel()

        condition = (R < 163) | (B < 163) | (G < 163)
        rmbkR = R[condition]
        rmbkG = G[condition]
        rmbkB = B[condition]
        plt.subplot(231)
        sns.kdeplot(rmbkR, cumulative=False, fill=False)
        plt.title('Red Grayscale')
        
        plt.subplot(232)
        sns.kdeplot(rmbkG, cumulative=False, fill=False)
        plt.title('Green Grayscale')
        
        plt.subplot(233)
        sns.kdeplot(rmbkB, cumulative=False, fill=False)
        plt.title('green Grayscale')

    # plot defective feature
    for i in random_numbers:
        singleBean = defectiveBeanImages[i]
        R = singleBean[:, :, 2].ravel()
        G = singleBean[:, :, 1].ravel()
        B = singleBean[:, :, 0].ravel()

        condition = (R < 163) | (B < 163) | (G < 163)
        rmbkR = R[condition]
        rmbkG = G[condition]
        rmbkB = B[condition]
        plt.subplot(234)
        sns.kdeplot(rmbkR, cumulative=False, fill=False)
        plt.title('Red Grayscale')
        
        plt.subplot(235)
        sns.kdeplot(rmbkG, cumulative=False, fill=False)
        plt.title('Green Grayscale')
        
        plt.subplot(236)
        sns.kdeplot(rmbkB, cumulative=False, fill=False)
        plt.title('green Grayscale')
    plt.tight_layout()
    plt.show()