#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 23:06
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : demo_BM.PY

import cv2
from utils import *
from calibrator import Calibrator
from stereo_matcher import StereoMatcher
import matplotlib.pyplot as plt

# for not rectified images
# stereo calibration
calibrator = Calibrator(images_dir="./data/calibration_images", border_size=(9, 6))
calibrator.calibrate()
calibrator.save("stereo_camera.json")

# stereo matching
matcher1 = StereoMatcher(methods="BM", is_rectified=False, camera_params_file="stereo_camera.json")
imgL1 = cv2.imread("./data/calibration_images/left06.jpg", flags=0)
imgR1 = cv2.imread("./data/calibration_images/right06.jpg", flags=0)
print(len(imgR1.shape))
rectifiedImgL1, rectifiedImgR1 = rectifyImagePair(imgL1, imgR1, cameraParams=json_load("stereo_camera.json"))
rectifiedImgPair = np.concatenate((rectifiedImgL1, rectifiedImgR1), axis=1)
rectifiedImgPair[::20, :] = 0
disp1 = matcher1.predict(imgL1, imgR1)
disp1 = cv2.normalize(disp1, disp1, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow("rectifiedImgP", rectifiedImgPair)
cv2.imshow("disp1", disp1)


# for rectified images
# stereo matching
matcher2 = StereoMatcher(methods="BM", is_rectified=True)
imgL2 = cv2.imread("./data/stereo_pairs/tsukuba_l.png", flags=0)
imgR2 = cv2.imread("./data/stereo_pairs/tsukuba_r.png", flags=0)
disp2 = matcher2.predict(imgL2, imgR2)
disp2 = cv2.normalize(disp2, disp2, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
imgPair = np.concatenate((imgL2, imgR2), axis=1)
imgPair[::20, :] = 0
cv2.imshow("imgPair", imgPair)
cv2.imshow("disp2", disp2)
cv2.waitKey()