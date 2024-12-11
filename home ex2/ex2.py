import cv2
import numpy as np


def display_image(img):
    cv2.imshow("mat", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


original_img = cv2.imread("baffalo.png", 0)

i = 7
a = cv2.bilateralFilter(original_img, i, i * 2, i / 2)
# display_image(a)

_, b = cv2.threshold(a, 128, 255, cv2.THRESH_BINARY_INV)
display_image(b)

results = []
dirs = [([-1, 1], [-1, 1]), ([-1, 1], [1, -1]), ([1, -1], [-1, 1]), ([1, -1], [1, -1])]


for dir in dirs:
    x1 = np.array(dir[0])
    x2 = np.array(dir[1]).reshape(1, len(dir[1]))
    res = cv2.filter2D(cv2.filter2D(b, -1, x1), -1, x2)
    results.append(res)


color_image = cv2.cvtColor(
    original_img,
    cv2.COLOR_BGR2RGB,
)

for res in results:
    color_image[res == 255] = [0, 0, 255]

display_image(color_image)


