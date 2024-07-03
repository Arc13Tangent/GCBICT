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

    attributesNames = ['R_mean', 'R_std', 'G_mean', 'G_std', 'B_mean', 'B_std']
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
    return attributesNames, attributesValues, allLabel

def allData(inputPath_defective, inputPath_qualified):
    
    attributesNames_d, attributesValues_d,labels_d = getData(inputPath_defective, 0)
    
    attributesNames_q, attributesValues_q,labels_q = getData(inputPath_qualified, 1)
    
    attributesNames = attributesNames_d
    attributesValues = np.array(attributesValues_d + attributesValues_q)
    labels = np.array(labels_d + labels_q)
    
    return attributesNames, attributesValues,labels


# Create dataset via class
def createCoffeeDataset(inputPath_defective, inputPath_qualified):

    attributesNames, attributesValues, labels = allData(inputPath_defective, inputPath_qualified)
    
    CoffeeDataset = beansData(attributesNames, attributesValues, labels)
    
    return CoffeeDataset


# Create training set and test set 
def trainAndTest(CoffeeDataset):

    trainIndices, testIndices = (
        train_test_split(range(len(CoffeeDataset.AttributesValues)), \
                         test_size=0.1,random_state=0, stratify = CoffeeDataset.Labels) 
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

    joblib.dump(CoffeeModel, 'Model/coffee_model.pkl')
    print(f"\n  The best parameter is {CoffeeModel.best_params_}") 
    print(f"\n  Test scores: {CoffeeModel.cv_results_['mean_test_score']}") 

    predictLabel = CoffeeModel.predict(testData)
    output_format = "{:.2f}"
    print("\n  Accuracy: {}%".format(output_format.format(100*metrics.accuracy_score(testLabel, predictLabel))))
    print("  F1 score:",metrics.f1_score(testLabel, predictLabel))
    matrix = confusion_matrix(testLabel, predictLabel)

    # Confusion Matrix
    print("  Confusion Matrix:")
    print("\t\t\tPredicted\tPredicted")
    print("\t\t\tdefective\tqualified")
    print("  Actual defective \t{}\t\t{}".format(matrix[0, 0], matrix[0, 1]))
    print("  Actual qualified \t{}\t\t{}".format(matrix[1, 0], matrix[1, 1]))

    xticklabels = ['defective', 'qualified']
    yticklabels = ['defective', 'qualified']

    sns.set(font_scale=2.5,rc={'figure.figsize':(9.5,7)})
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['font.weight'] = 'bold'

    sns.heatmap(matrix, fmt = '', cmap="Blues", cbar=False, 
            yticklabels=yticklabels, xticklabels=xticklabels,
            annot_kws={"size": 40}, 
            annot=[[f"TN\n({matrix[0][0]:.0f})", f"FP\n({matrix[0][1]:.0f})"], [f"FN\n({matrix[1][0]:.0f})", f"TP\n({matrix[1][1]:.0f})"]]) 

    plt.xlabel('Predicted Values', fontsize=24)
    plt.ylabel('Actual Values', fontsize=24)
    plt.title('Confusion matrix for the model', fontsize=28, fontweight='bold')
    plt.show()
