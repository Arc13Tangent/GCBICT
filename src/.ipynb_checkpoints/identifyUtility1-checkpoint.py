import os
import cv2
import numpy as np
import joblib
from src.cut import cut
from src.compute import compute
from scipy.stats import norm

def predict(inputPath, coffeeModel):
    
    image = cv2.imread(inputPath)
    contours = cut(image)
    
    # For beans in image
    for contour in contours:
        if cv2.contourArea(contour) > 100: # large enough

            x, y, w, h = cv2.boundingRect(contour) 
            singleBean = image[y:y+h, x:x+w]
            
            data = np.array(compute(singleBean))
            pred = coffeeModel.predict(data.reshape(1, -1))

            if pred == 0:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2) # green for qualified
            else:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2) # red for defective
                
    imageName = os.path.basename(inputPath)
    filename = 'Result/' + imageName
    cv2.imwrite(filename, image)