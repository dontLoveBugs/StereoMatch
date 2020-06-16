#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 13:13
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : utils.py


import cv2
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def json_save(filename, vars):
    with open(filename, "w") as fw:
        json.dump(vars, fw, cls=NumpyEncoder)


def json_load(filename):
    with open(filename, "r") as fr:
        vars = json.load(fr)
    for k, v in vars.items():
        vars[k] = np.array(v)
    return vars


def rectifyImagePair(imgL, imgR, cameraParams):
    """
    :param imgL: cv2.image(np.array)
    :param imgR: cv2.image(np.array)
    :param imageShape: py.tuple, (h, w)
    :param CameraParams: py.dict, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T
    :return: rectifiedImgL, rectifiedImgR: cv2.image(np.array)
    """
    assert imgL.shape == imgR.shape, "imgL.shape != imgR.shape."
    h, w = imgL.shape[:2]
    cameraMatrix1, distCoeffs1 = cameraParams["cameraMatrix1"], cameraParams["distCoeffs1"]
    cameraMatrix2, distCoeffs2 = cameraParams["cameraMatrix2"], cameraParams["distCoeffs2"]
    R, T = cameraParams["R"], cameraParams["T"]

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = \
        cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (w, h), R, T)

    map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (w, h), cv2.CV_16SC2)
    map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (w, h), cv2.CV_16SC2)

    rectifiedImgL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    rectifiedImgR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

    return rectifiedImgL, rectifiedImgR
