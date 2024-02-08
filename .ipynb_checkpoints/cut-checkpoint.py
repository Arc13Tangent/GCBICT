import cv2
import numpy as np
import joblib
from scipy.stats import norm

def cut(image):
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Convert to HSV to remove background
    backgroundMin = np.array([0, 0, 165])
    backgroundMax = np.array([180, 20, 255])
    backgroundMask = cv2.inRange(hsv, backgroundMin, backgroundMax) # Select the brackground (mark as ~0)
    rmbkResult = cv2.bitwise_and(image, image, mask=~backgroundMask) # Leave the foreground, background mark as 0
    grayResult = cv2.cvtColor(rmbkResult, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    _, binaryResult = cv2.threshold(grayResult, 1, 255, cv2.THRESH_BINARY) # mark ~0 elements as 255
    contours, _ = cv2.findContours(binaryResult, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

