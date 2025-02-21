import os

import cv2
import matplotlib.pyplot as plt

# import glob
import numpy as np
import scipy
import scipy.ndimage
import skimage
from PIL import Image


def display_image(img):
    cv2.imshow("mat", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


complete_image = Image.open("keyboard-1.jpg")
complete_image = complete_image.convert("L")

complete_image = np.asarray(complete_image, dtype="uint8") / 255.0
complete_image = 1 - complete_image
complete_imagcopy = complete_image

pieces_im = Image.open("keyboard-2.jpg")  # Requires a black background
gray_im = pieces_im.convert("L")
gray_im = np.asarray(gray_im, dtype="uint8") / 255.0
gray_im = 1 - gray_im

outh, outw = complete_image.shape
output_image = np.zeros((outh, outw))

binarized_image = np.zeros(gray_im.shape)
binarized_image[gray_im > 0.7] = 1.0

image = scipy.ndimage.binary_opening(
    binarized_image,
    structure=np.ones((3, 3)),
    iterations=1,
    output=None,
    origin=0,
)

image = scipy.ndimage.binary_closing(
    image, structure=np.ones((3, 3)), iterations=1, output=None, origin=0
)

cv_image = skimage.util.img_as_ubyte(image)

contours, hierarchy = cv2.findContours(
    cv_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

cv2.drawContours(cv_image, contours, -1, (255, 255, 0), 2)
plt.imshow(cv_image, cmap=plt.cm.gray)
plt.title("Pieces with contures")
plt.axis("off")
plt.show()

piece_array = []
for i in range(0, len(contours)):
    c = np.asarray(contours[i])
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(cv_image, (x, y), (x + w, y + h), (255, 255, 0), 2)
    crop_img = gray_im[y : y + h, x : x + w]
    piece_array.append(crop_img)
plt.imshow(cv_image, cmap=plt.cm.gray)
plt.axis("off")
plt.show()

# Initiate SIFT detector
sift = cv2.SIFT_create()

complete_imagcopy = skimage.util.img_as_ubyte(complete_imagcopy)
kp2, des2 = sift.detectAndCompute(complete_imagcopy, None)

bf = cv2.BFMatcher()
os.makedirs("./output", exist_ok=True)
i = 0
for v in piece_array:
    i += 1
    v = skimage.util.img_as_ubyte(v)
    kp1, des1 = sift.detectAndCompute(v, None)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 5

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(
            -1, 1, 2
        )

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = v.shape
        pts = np.float32(
            [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
        ).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        complete_imagcopy = cv2.polylines(
            complete_imagcopy, [np.int32(dst)], True, 255, 3, cv2.LINE_AA
        )

    else:
        matchesMask = None
    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matchesMask,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    img3 = cv2.drawMatches(v, kp1, complete_imagcopy, kp2, good, None, flags=2)

    plt.imshow(img3)
    plt.axis("off")
    plt.savefig(f"./output/match {i}")

plt.imshow(complete_imagcopy, cmap="gray")
plt.title("final image")
plt.axis("off")
plt.savefig("test.jpg")
plt.show()
