import time
from colorama import Fore, Style
from src.trainingUtilities2 import createCoffeeDataset, trainModel
import joblib
from src.identifyUtility2 import predict

def function2_Training():
    while True:
        command = input("  ➡ Choose multiple sites [C] / Start training [T] / Exit [E]: ")
        # Choose multiple sites [C]
        if command in ['C', 'c']:
            while True:
                site_num = input('How many sites/varities you have? ')
                if site_num in ['E', 'e', 'Exit', 'exit', 'EXIT']:
                    break
                try:
                    site_num = int(site_num)
                    if site_num <= 1:
                        raise ValueError()
                        
                    inputPath = []
                    for i in range(0, site_num):
                        path = input('    🗀 Directory for site/varity {}:\n      '.format(i+1))
                        inputPath.append(path)
                    print(inputPath)
                    break
                except:
                    print('    ‼  Please check the number (should be greater than 1).')
                
        # Start training [T]
        elif command in ['T', 't']:
            try:
                print('  Progress: ', end = '')
                CoffeeDataset = createCoffeeDataset(inputPath)
                trainModel(CoffeeDataset)
                print('  Done!')
            except:
                print('    ‼  Please check the directories.')
        elif command in ['E', 'e', 'Exit', 'exit', 'EXIT']:
            break
        else:
            print('    ‼︎  Wrong command, please try again.')

def function2_Identify():
    inputPath = input("  🗏 Choose image file:\n    ")
    try:
        coffeeModel = joblib.load('Model/coffee_model_multi.pkl')
        predict(inputPath, coffeeModel)
        print('  Done!')
    except:
        print('    ‼ Please check the directory or model.')  