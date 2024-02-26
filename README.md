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
![interface](https://github.com/Arc13Tangent/GCBICT/assets/117557116/66a0c720-ce80-49c2-a0e7-1fdd08ddde61)


## Example: Qualified-Defective separator
## Training
Users type $\texttt{Q}$ to select the Qualified-Defective separator. Then type $\texttt{Q}$
and $\texttt{D}$ to determine the directories of folders that contain qualified beans
and defective beans, respectively. Each folder can contain multiple images of
beans.

<center>
    <blockquote class="imgur-embed-pub" lang="en" data-id="ZaLMyu1"><a href="https://imgur.com/ZaLMyu1">View post on imgur.com</a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>
    <img src="https://imgur.com/ZaLMyu1" alt="Each folder consists of multiple images of beans"> 
    Each folder consists of multiple images of beans
</center>
<br> 

After the above steps, type $\texttt{T}$ to obtain the model of support vector machine (SVM) type. The model file is placed in the “Model” folder.
<center>
    <img src="https://hackmd.io/_uploads/BkCy5tYhp.jpg" alt="The whole workflow of training stage"> 
    The whole workflow of training stage
</center>
<br> 

<center>
    <img src="https://hackmd.io/_uploads/rkSlcFYhT.png" alt="The confusion matrix of the model generated by GCBICT in this example"> 
    The confusion matrix of the model generated by GCBICT in this example
</center>
<br> 

## Identification
After the training stage, users type $\texttt{E}$ to leave the training mode, and then type $\texttt{I}$ to select the identification mode. In the identification mode, choose the folder containing the images of unevaluated beans as input and use the model obtained in the training stage to identify the qualified and defective beans in the chosen folder.

<center>
    <img src="https://hackmd.io/_uploads/r1Tmcttha.jpg" alt="The identification mode demonstration"> 
    The identification mode demonstration
</center>
<br> 

## Result
<center>
    <img src="https://hackmd.io/_uploads/r1HV5FKha.jpg" alt="The identification result"> 
</center>
<br> 
