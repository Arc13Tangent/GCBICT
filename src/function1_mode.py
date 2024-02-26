import time
from colorama import Fore, Style
from src.trainingUtilities1 import createCoffeeDataset, trainModel
import joblib
from src.identifyUtility1 import predict
from src.drawFeature import drawFeature

def function1_Training():
    while True:
        command = input("  ➡ Choose qualified [Q] / Choose defective [D] / Start training [T] / Exit [E]: ")
        # Choose qualified [Q]
        if command in ['Q', 'q']:
            inputPath_qualified = input('    🗀 Directory for qualified:\n      ')
        # Choose defective [D]
        elif command in ['D', 'd']:
            inputPath_defective = input('    🗀 Directory for defective:\n      ')
        # Start training [T]
        elif command in ['T', 't']:
            try:
                print('  Progress: ', end = '')
                CoffeeDataset = createCoffeeDataset(inputPath_defective, inputPath_qualified)
                drawFeature(inputPath_defective, inputPath_qualified)
                trainModel(CoffeeDataset)
                print('  Done!')
            except:
                print('    ‼  Please check the directories.')
        elif command in ['E', 'e', 'Exit', 'exit', 'EXIT']:
            break
        else:
            print('    ‼︎  Wrong command, please try again.')

def function1_Identify():
    inputPath = input("  🗏 Choose image file:\n    ")
    try:
        coffeeModel = joblib.load('Model/coffee_model.pkl')
        predict(inputPath, coffeeModel)
        print('  Done!')
    except:
        print('    ‼ Please check the directory or model.')  