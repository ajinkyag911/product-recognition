import cv2
import numpy as np
import matplotlib.pyplot as plt

#img = cv2.imread('dataset/models/0.jpg',cv2.IMREAD_GRAYSCALE)

img_rgb = cv2.imread('dataset/scenes/m5.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

template = cv2.imread('dataset/models/25.jpg',0)
w, h = template.shape[::-1]

scale_percent = 50 # percent of original size
width = int(template.shape[1] * scale_percent / 100)
height = int(template.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv2.resize(template, dim, interpolation = cv2.INTER_AREA)

res = cv2.matchTemplate(img_gray,resized,cv2.TM_CCOEFF_NORMED)
threshold = 0.5
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 5)

cv2.imshow('Detected',img_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()