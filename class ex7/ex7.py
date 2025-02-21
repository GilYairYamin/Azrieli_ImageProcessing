import cv2
import matplotlib.pyplot as plt
import numpy as np


def display_image(img, title="image"):
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


original_image = cv2.imread("pic.png", cv2.IMREAD_GRAYSCALE)
# display_image(original_image)
display_image(original_image)

F = np.fft.fft2(original_image)
S = np.abs(F)
print(S.shape)

Fc = np.fft.fftshift(F)
Ss = np.abs(Fc)

# display_image(Ss)

width = 195
height = 193

thickness = 30
mask1 = np.ones((width, height))
mask2 = np.ones((width, height))
mask3 = np.zeros((width, height))
mask4 = np.zeros((width, height))
mask5 = np.zeros((width, height))
mask6 = np.ones((width, height))
mask7 = np.ones((width, height))
mask8 = np.zeros((width, height))

mask1[:, (height // 2 - thickness) : (height // 2 + thickness)] = 0
mask2[(width // 2 - thickness) : (width // 2 + thickness),] = 0
mask3[:, (height // 2 - thickness) : (height // 2 + thickness)] = 1
mask4[(width // 2 - thickness) : (width // 2 + thickness),] = 1
mask5[
    (width // 2 - thickness) : (width // 2 + thickness) :,
    (height // 2 - thickness) : (height // 2 + thickness),
] = 1
mask6[
    (width // 2 - thickness) : (width // 2 + thickness) :,
    (height // 2 - thickness) : (height // 2 + thickness),
] = 0

mask7[:, (height // 2 - thickness) : (height // 2 + thickness)] = 0
mask7[(width // 2 - thickness) : (width // 2 + thickness),] = 0

mask8[:, (height // 2 - thickness) : (height // 2 + thickness)] = 1
mask8[(width // 2 - thickness) : (width // 2 + thickness),] = 1

masks = [mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8]

for mask in masks:
    res1 = Fc * mask
    res2 = np.fft.ifftshift(res1)
    res3 = np.fft.ifft2(res2)
    res4 = np.real(res3)
    display_image(mask)
    display_image(res4)
