import cv2
import numpy as np
import matplotlib.pyplot as plt


def Histogram_Equalization(src, z_max=255,):
    h, w = src.shape
    sum_h = 0
    S = h * w
    out = np.zeros((h, w), dtype=np.uint8)
    for i in range(0, 255):
        ind = np.where(src == i)
        sum_h += len(src[ind])
        z_prime = z_max / S * sum_h
        out[ind] = z_prime
    return out


if __name__ == '__main__':
    src = cv2.imread("./img/1.png") # [500, 500, 3]
    dst = Histogram_Equalization(src[:, :, 0])
    cv2.imwrite('./img/res_1.png', dst)