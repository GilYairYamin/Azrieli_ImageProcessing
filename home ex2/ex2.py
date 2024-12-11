import cv2
import numpy as np


def display_image(img):
    cv2.imshow("mat", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


original_img = cv2.imread("baffalo.png", 0)

i = 7

biliteral_filtered = cv2.bilateralFilter(original_img, i, i * 2, i / 2)
_, thresholded = cv2.threshold(biliteral_filtered, 128, 255, cv2.THRESH_BINARY_INV)

results = []


dirs = [([-1, 1], [-1, 1]), ([-1, 1], [1, -1]), ([1, -1], [-1, 1]), ([1, -1], [1, -1])]

# create 4 directions for which we find corners.
# This doesn't work perfectly using the buffalo image provided.
for dir in dirs:
    x1 = np.array(dir[0])
    x2 = np.array(dir[1]).reshape(1, len(dir[1]))
    res = cv2.filter2D(cv2.filter2D(thresholded, -1, x1), -1, x2)
    results.append(res)


# Color the image where we found corners
color_image = cv2.cvtColor(
    original_img,
    cv2.COLOR_BGR2RGB,
)

for res in results:
    color_image[res == 255] = [0, 0, 255]

display_image(color_image)
cv2.imwrite('buffalo_corners.jpg', color_image)

# Honestly, I don't even know how to approach the question of calculating the area of the square using this data.
# And I don't think I did this well.