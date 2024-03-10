# GCBICT: Green Coffee Bean Identification Command-line Tool
> Author: Shu-Min Tan, Shih-Hsun Hung, and Je-Chiang Tsai, National Tsing Hua University.\
> Last update: Mar 10, 2024

## Getting started
To clone the source code from GitHub, type 
```console
$ git clone https://github.com/Arc13Tangent/GCBICT.git
```
in the command line, then navigate to the GCBICT folder and type 
```console
$ cd GCBICT
$ pip install -r Requirements.txt
```
to install all the required packages specified in the ```Requirements.txt```.
<div align="center">
<img src="https://i.imgur.com/zrBsxc5.jpeg" alt="Install all the requirements specified in the Requirements.txt">
</div>

## Software architecture
The command-line tool GCBICT consists of two functions: (1) the Qualified-Defective Separator and (2) the Mixed Separator.
The Qualified-Defective Separator allows users to choose the images of multiple qualified and defective green coffee beans in the training stage, then train the machine learning model to obtain model parameters, and finally apply the trained model to the image of unevaluated beans to identify their quality. The Mixed Separator allows users to choose the images of a group of qualified beans with multiple growing sites/varieties, then train the machine learning model to obtain model parameters, and finally apply the trained model to the images of a group of qualified beans with multiple growing sites/varieties to identify their growing sites/varieties. Users can view the model details and identification results in the ```Model``` and ```Result``` folders. The algorithms that use our color characteristics of the seat coat of green coffee beans in GCBICT are patented.
<div align="center">
<img src="https://i.imgur.com/pXUZgFl.png" alt="The GCBICT architecture">
<p>The GCBICT architecture</p>
</div>
<div align="center">
<img src="https://i.imgur.com/TvBkyDS.png" alt="The GCBICT architecture">
<p>Our recently
discovered intrinsic color characteristics of the seat coat of green coffee beans. Panels (A), (B), (C) present the statistical features of the chosen 30 qualified beans, while (D), (E), (F) give the statistical features of the chosen 30 defective beans.</p>
</div>

## Using GCBICT
To run GCBICT, open the terminal, navigate to the GCBICT folder, and input the command ```python main.py```:
```console
$ cd GCBICT
$ python main.py # or python3 main.py
```
<div align="center">
<img src="https://i.imgur.com/LJ7UPyF.jpeg" alt="The GCBICT interface">
</div>

## Example 1: Identification of qualified beans
### Training
Users type ```Q``` to select the Qualified-Defective Separator. 
Then, respectively, type ```Q``` and ```D``` to input the folders' paths where the images of the qualified and defective beans of the training set are stored.
<div align="center">
    <img src="https://imgur.com/GFhqhlQ.png" alt="Each folder consists of multiple images of beans"> 
    <p>Each folder consists of multiple images of beans</p>
</div>
<br> 

After the above steps, 
type ```T``` to obtain the model of support vector machine (SVM) type. The model file is placed in the ```Model``` folder.
<div align="center">
    <img src="https://imgur.com/4sitdy5.jpg" alt="The whole workflow of training stage"> 
    <p>The whole workflow of the training stage</p>
</div>
<br> 

The computed color characteristics of beans' image and the confusion matrix of the machine learning model are shown in the following graphs.
<div align="center">
    <img src="https://i.imgur.com/w1x1GBq.png" alt="The computed color characteristics in this example"> 
    <p>The computed color characteristics in this example</p>
</div>
<br> 

<div align="center">
    <img src="https://imgur.com/5f8Sg7w.png" alt="The confusion matrix of the model generated by GCBICT in this example"> 
    <p>The confusion matrix of the model generated by GCBICT in this example</p>
</div>
<br> 

### Identification
After the training stage, users type ```E``` to leave the training mode, and then type ```I``` to select the identification mode. 
In the identification mode, 
input the path of the folder where the images of unevaluated beans are stored.
Then the GCBICT uses the model obtained in the training stage to identify the qualified and defective beans of the test set.
<div align="center">
    <img src="https://imgur.com/sFDIEoN.jpg" alt="The identification mode demonstration"> 
    <p>The identification mode demonstration</p>
</div>
<br> 

### Result
<div align="center">
    <img src="https://imgur.com/fszW6BS.png" alt="The identification result"> 
</div>
<br>  
    
## Example 2: Identification of the growing site of beans
### Training
The user first types ```E``` to leave the Qualified-Defective Separator and then types ```M``` to select the Mixed Separator.
Next, type ```T``` to select the training mode. 
Then type ```C``` to specify the number of growing sites, and input the path of the folder where the images of beans of each growing site in the training set are stored. 
<div align="center">
    <img src="https://imgur.com/A0BL2Ho.jpg" alt="The Mixed separator"> 
    <p>The demonstration of the Mixed Separator</p>
</div>
<br>  

Next, type ```T``` to obtain the model of support vector machine (SVM) type. 
The model file is placed in the “Model” folder
and the confusion matrix of the machine learning model is shown in the following graph.
<div align="center">
    <img src="https://imgur.com/cWGtuPm.png" alt="The confusion matrix"> 
    <p>The confusion matrix of the model generated by the Mixed Separator</p>
</div>
<br> 

### Identification
After the training stage, users type ```E``` to leave the training mode 
and then type ```I``` to select the identification mode. In the identification mode, input the path of the folder where the images of unevaluated beans are stored. Then the GCBICT uses the model obtained in the training stage to identify the growing sites of beans in the test set. The result is placed in the “Result” folder.
### Result
<div align="center">
    <img src="https://imgur.com/SOKnTjO.jpg" alt="The identification result"> 
    <p>The identification result for beans from different growing sites.
    The bean with the green (red and blue) box is identified as being from site 1 (site 2 and site 3, respectively).</p>
</div>
<br> 
