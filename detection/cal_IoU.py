#coding=UTF-8
import numpy as np


def cal_IoU(boxA, boxB):
    area_A = (boxA[:, 2] - boxA[:, 0] + 1) * (boxA[:, 3] - boxA[:, 1] + 1)
    area_B = (boxB[:, 2] - boxB[:, 0] + 1) * (boxB[:, 3] - boxB[:, 1] + 1)
    # 相交部分坐标
    xx1 = np.maximum(boxA[:, 0], boxB[:, 0])
    yy1 = np.maximum(boxA[:, 1], boxB[:, 1])
    xx2 = np.minimum(boxA[:, 2], boxB[:, 2])
    yy2 = np.minimum(boxA[:, 3], boxB[:, 3])
    # 相交部分面积
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    # IoU
    IoU = inter / (area_A + area_B - inter)
    return IoU

def cal_GIoU(boxA, boxB):
    area_A = (boxA[:, 2] - boxA[:, 0] + 1) * (boxA[:, 3] - boxA[:, 1] + 1)
    area_B = (boxB[:, 2] - boxB[:, 0] + 1) * (boxB[:, 3] - boxB[:, 1] + 1)
    # 相交部分坐标
    xx1 = np.maximum(boxA[:, 0], boxB[:, 0])
    yy1 = np.maximum(boxA[:, 1], boxB[:, 1])
    xx2 = np.minimum(boxA[:, 2], boxB[:, 2])
    yy2 = np.minimum(boxA[:, 3], boxB[:, 3])
    # 相交部分面积
    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    # IoU
    IoU = inter / (area_A + area_B - inter)
    return IoU


if __name__ == '__main__':
    boxA = np.array([[5, 5, 10, 10], [5, 5, 10, 10], [5, 5, 10, 10], [5, 5, 10, 10]])
    boxB = np.array([[5, 5, 10, 10], [7, 7, 15, 15], [1, 1, 8, 8], [6, 6, 9, 9]])
    IoU = cal_IoU(boxA, boxB)
    print("IoU={}".format(IoU))

    GIoU = cal_GIoU(boxA, boxB)
    print("GIoU={}".format(GIoU))
