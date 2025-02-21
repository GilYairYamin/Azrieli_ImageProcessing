import cv2
import numpy as np
from matplotlib import pyplot as plt


def display_image(img):
    cv2.imshow("mat", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


original_img = cv2.imread("rice-shaded.tif", 0)
# display_image(original_img)

# equ = cv2.equalizeHist(original_img)
# display_image(equ)



# display_image(test1)


clahe = cv2.createCLAHE(0.3, (8, 8))
cl1 = clahe.apply(original_img)
display_image(cl1)

# test = cv2.adaptiveThreshold(
#     cl1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 1
# )

test = cv2.threshold(cl1, 128, 255, cv2.THRESH_BINARY)
display_image(test[1])

kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
test1 = cv2.morphologyEx(test[1], cv2.MORPH_OPEN, kernal)
display_image(test1)