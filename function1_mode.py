import time
from colorama import Fore, Style
from trainingUtilities1 import createCoffeeDataset, trainModel
import joblib
from identifyUtility1 import predict

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
                tic = time.time()
                print('  Progress: ', end = '')
                CoffeeDataset = createCoffeeDataset(inputPath_defective, inputPath_qualified)
                trainModel(CoffeeDataset)
                elapsed = time.time() - tic
                print('  Done!\n  Elapsed time: {} sec.'.format(elapsed))
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
        tic = time.time()
        predict(inputPath, coffeeModel)
        elapsed = time.time() - tic
        print('  Done! Elapsed time: {} sec.'.format(elapsed))
    except:
        print('    ‼ Please check the directory or model.')  