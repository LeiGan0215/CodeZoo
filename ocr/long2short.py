# coding=utf-8
import cv2
import numpy as np
import os


def vProject(binary):
    h, w = binary.shape
    w_w = [0]*w
    for i in range(w):
        for j in range(h):
            if binary[j, i ] == 0:
                w_w[i] += 1
    return w_w


def segmentation(img, res_path='./img/seg_1', seg_num=10):
    # 1.二值化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # 2.按列统计像素数目
    h, w = th.shape
    position = []
    w_w = vProject(th)
    # 3.字符切割
    wstart, w_start, w_end = 0, 0, 0
    for j in range(len(w_w)):
        if w_w[j] > 3 and wstart == 0:
            w_start = j
            wstart = 1
        if w_w[j] == 0 and wstart == 1:
            w_end = j
            wstart = 0
            position.append([w_start, w_end])

    # 4.每seg_num个字符进行切割
    start = position[0][0]
    end = position[0][1]
    n=1
    for i in range(0, len(position), seg_num):
        if i > 0 and i % seg_num == 0:
            end = position[i][1]
            img_i = img[:, start : end]
            name = os.path.join(res_path, str(n)+'.png')
            cv2.imwrite(name, img_i)
            start = end
            n+=1
    # 5.最后一段
    end = position[-1][1]
    img_last = img[:,start:end]
    name = os.path.join(res_path, str(n) + '.png')
    cv2.imwrite(name, img_last)


if __name__ == '__main__':
    img = cv2.imread('./img/1.jpg')
    segmentation(img, res_path='./img/seg_1', seg_num=10)