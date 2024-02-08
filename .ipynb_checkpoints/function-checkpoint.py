from function1_mode import function1_Training, function1_Identify
from function2_mode import function2_Training, function2_Identify
from colorama import Fore, Style

def function1():
    while True:
        mode = input('  〖Q-D〗Training [T] / Identify [I] / Exit Q-D separator [E]: ')
        if mode in ['T', 't']:
            function1_Training()
        elif mode in ['I', 'i']:
            function1_Identify()
        elif mode in ['E', 'e', 'Exit', 'exit', 'EXIT']:
            break
        else:
            print('    ‼ Wrong command, please try again.')

def function2():
    while True:
        mode = input('  〖M〗  Training [T] / Identify [I] / Exit Mixed separator [E]: ')
        if mode in ['T', 't']:
            function2_Training()
        elif mode in ['I', 'i']:
            function2_Identify()
        elif mode in ['E', 'e', 'Exit', 'exit', 'EXIT']:
            break
        else:
            print('    ‼ Wrong command, please try again.')

def function():
     while True:
         function = input('𝓒 Qualified-Defective separator [Q] / Mixed separator [M] / Exit GCBICT [E]: ')
         if function in ['Q', 'q']:
             function1()
         elif function in ['M', 'm']:
             function2()
         elif function in ['E', 'e', 'Exit', 'exit', 'EXIT']:
             print('Goodbye!')
             break
         else:
             print('    ‼ Wrong command, please try again.')