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
import matplotlib as mpl
import joblib
from src.cut import cut
from src.compute import compute

from sklearn.model_selection import GridSearchCV 


# Define class for coffee beans data
class beansData:
    def __init__(self, attributesNames, attributesValues, labels):

        self.AttributesNames = attributesNames # AttributesNames: call of attributes
        self.AttributesValues = attributesValues # AttributesValues: data of beans
        self.Labels = labels # Labels: labels for beans


# Read img and generate data
def getData(inputPath, label):

    attributesValues = []
    allLabel = []
    
    imgFiles = glob.glob(os.path.join(inputPath, '*.jpg'))
    # img: contains several beans
    for img in imgFiles:
        image = cv2.imread(img)
        beans = cut(image)
        for bean in beans:
            if cv2.contourArea(bean) > 100: # large enough
                x, y, w, h = cv2.boundingRect(bean) 
                singleBean = image[y:y+h, x:x+w]
                singleData = compute(singleBean)
                attributesValues.append(singleData)
                allLabel.append(label)
        
    return attributesValues, allLabel


def allData(inputPath):

    attributesNames = ['R_mean', 'R_std', 'G_mean', 'G_std', 'B_mean', 'B_std']
    attributesValues = []
    allLabels = []
    
    for i in range(0, len(inputPath)):
        values, labels = getData(inputPath[i], i+1)
        attributesValues = attributesValues + values
        allLabels = allLabels + labels
        
    return attributesNames, np.array(attributesValues), np.array(allLabels)


# Create dataset via class
def createCoffeeDataset(inputPath):

    attributesNames, attributesValues, labels = allData(inputPath)
    
    CoffeeDataset = beansData(attributesNames, attributesValues, labels)
    
    return CoffeeDataset


# Create training set and test set 
def trainAndTest(CoffeeDataset):

    trainIndices, testIndices = (
        train_test_split(range(len(CoffeeDataset.AttributesValues)), \
                         test_size=0.6,random_state=0, stratify = CoffeeDataset.Labels) 
    )

    trainData = CoffeeDataset.AttributesValues[trainIndices]
    trainLabel = CoffeeDataset.Labels[trainIndices]
    
    testData = CoffeeDataset.AttributesValues[testIndices]
    testLabel = CoffeeDataset.Labels[testIndices]
    
    return trainIndices, testIndices, trainData, testData, trainLabel, testLabel


# Model training
def trainModel(CoffeeDataset):

    trainIndices, testIndices, trainData, testData, trainLabel, testLabel = trainAndTest(CoffeeDataset)

    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

    CoffeeModel = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 1) 
    #CoffeeModel = svm.SVC(kernel='linear')
    
    CoffeeModel.fit(trainData, trainLabel)
    joblib.dump(CoffeeModel, 'Model/coffee_model_multi.pkl')
    print(f"\n  The best parameter is {CoffeeModel.best_params_}") 
    print(f"\n  Test scores: {CoffeeModel.cv_results_['mean_test_score']}") 
    
    predictLabel = CoffeeModel.predict(testData)
    output_format = "{:.2f}"
    print("\n  Accuracy: {}%".format(output_format.format(100*metrics.accuracy_score(testLabel, predictLabel))))
    print("  F1 score:", metrics.f1_score(testLabel, predictLabel, average = 'weighted'))
    matrix = confusion_matrix(testLabel, predictLabel)

    print("  Confusion Matrix:")
    num = matrix.shape[0]
    for i in range(0, num):
        if i == num-1:
            print("Predicted")
        elif i == 0:
            print("\t   Predicted ", end='')
        else:
            print("Predicted ", end='')
            
    for i in range(0, num):
        if i == num-1:
            print("      {}".format("{:4d}".format(i+1)))
        elif i == 0:
            print("\t\t{}".format("{:4d}".format(i+1)), end='')
        else:
            print("      {}".format("{:4d}".format(i+1)), end='')

    for i in range(0, num):
        if i == num-1:
            print("----------")
        elif i == 0:
            print("  ------------------", end='')
        else:
            print("----------", end='')
            
    for i in range(0, num):
        for j in range(0, num):
            if j == num-1:
                print("      {}".format("{:4d}".format(matrix[i, j])))
            elif j == 0:
                print("  Actual {}\t{}".format(i+1, "{:4d}".format(matrix[i, j])), end='')
            else:
                print("      {}".format("{:4d}".format(matrix[i, j])), end='')

    xticklabels = [f'Site {i+1}' for i in range(num)]
    yticklabels = [f'Site {i+1}' for i in range(num)]
    sns.set(font_scale=2.5, rc={'figure.figsize':(9.5,7)})
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['font.weight'] = 'bold'
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, 
                yticklabels=yticklabels, xticklabels=xticklabels, 
                vmax=200, annot_kws={"size": 40})
    plt.xlabel('Predicted Values', fontsize=24)
    plt.ylabel('Actual Values', fontsize=24)
    plt.title('Confusion matrix for the model', fontsize=28, fontweight='bold')
    plt.show()