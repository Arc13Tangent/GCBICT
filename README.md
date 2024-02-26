# GCBICT: Green Coffee Bean IDentification Pipeline
> Author: Shu-Min Tan, Shih-Hsun Hung, and Je-Chiang Tsai, National Tsing Hua University.
> Last update: Feb 4, 2024

## Getting started
To install all the required packages, open the terminal, navigate to ```GCBICT``` and type ```pip install -r Requirements.txt``` in the command line:
```console
$ cd GCBICT
$ pip install -r Requirements.txt
```

## Using GCBICT
Open the terminal, navigate to ```GCBICT``` and type ```python main.py```:
```console
$ cd GCBICT
$ python main.py # or python3 main.py
```
<div align="center">
<img src="https://imgur.com/xYp3qFt.jpg" alt="The GCBICT interface">
</div>

## Example: Qualified-Defective separator
## Training
Users type $\texttt{Q}$ to select the Qualified-Defective separator. Then type $\texttt{Q}$
and $\texttt{D}$ to determine the directories of folders that contain qualified beans
and defective beans, respectively. Each folder can contain multiple images of
beans.

<div align="center">
    <img src="https://imgur.com/ZaLMyu1.png" alt="Each folder consists of multiple images of beans"> 
    <p>Each folder consists of multiple images of beans</p>
</div>
<br> 

After the above steps, type $\texttt{T}$ to obtain the model of support vector machine (SVM) type. The model file is placed in the “Model” folder.
<div align="center">
    <img src="https://imgur.com/Ve8bUXZ.jpg" alt="The whole workflow of training stage"> 
    <p>The whole workflow of the training stage</p>
</div>
<br> 

<div align="center">
    <img src="https://imgur.com/8V3y6in.png" alt="The confusion matrix of the model generated by GCBICT in this example"> 
    <p>The confusion matrix of the model generated by GCBICT in this example</p>
</div>
<br> 

## Identification
After the training stage, users type $\texttt{E}$ to leave the training mode, and then type $\texttt{I}$ to select the identification mode. In the identification mode, choose the folder containing the images of unevaluated beans as input and use the model obtained in the training stage to identify the qualified and defective beans in the chosen folder.

<div align="center">
    <img src="https://imgur.com/25TkBbX.jpg" alt="The identification mode demonstration"> 
    <p>The identification mode demonstration</p>
</div>
<br> 

## Result
<center>
    <img src="https://imgur.com/vHAGCNy.jpg" alt="The identification result"> 
</center>
<br> 
