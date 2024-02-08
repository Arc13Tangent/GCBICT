import os
import cv2
import numpy as np
import joblib
from cut import cut
from compute import compute
from scipy.stats import norm
import random

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1
text_color = (0, 0, 0)

def predict(inputPath, coffeeModel):
    num = len(coffeeModel.classes_)
    color_tuple = []
    for i in range(0, num):
        if i == 0:
            color_tuple.append((0,255,0))
        elif i == 1:
            color_tuple.append((0,0,255))
        elif i == 2:
            color_tuple.append((255,0,0))
        else:
            color_tuple.append((random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)))
    
    image = cv2.imread(inputPath)
    contours = cut(image)
    
    # For beans in image
    for contour in contours:
        if cv2.contourArea(contour) > 100: # large enough

            x, y, w, h = cv2.boundingRect(contour) 
            singleBean = image[y:y+h, x:x+w]
            
            data = np.array(compute(singleBean))
            pred = coffeeModel.predict(data.reshape(1, -1))

            cv2.rectangle(image, (x, y), (x+w, y+h), color_tuple[pred[0]-1], 1) 
            cv2.putText(image, str(pred[0]), (x+w, y+h), font, font_scale, text_color, font_thickness)
                
    imageName = os.path.basename(inputPath)
    filename = 'Result/' + imageName
    cv2.imwrite(filename, image)