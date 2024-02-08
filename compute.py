import cv2
import numpy as np
import joblib
from scipy.stats import norm

def compute(coffeeImg):

    B = coffeeImg[:, :, 0]  
    G = coffeeImg[:, :, 1]  
    R = coffeeImg[:, :, 2] 

    condition = (R < 163) | (B < 163) | (G < 163)
    rmbkR = R[condition]
    rmbkG = G[condition]
    rmbkB = B[condition]

    flatDataR = rmbkR.flatten()
    flatDataG = rmbkG.flatten()
    flatDataB = rmbkB.flatten()

    r  = norm.fit(flatDataR)
    g  = norm.fit(flatDataG)
    b  = norm.fit(flatDataB)
    
    return [r[0], r[1], g[0], g[1], b[0], b[1]]