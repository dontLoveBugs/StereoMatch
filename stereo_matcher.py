#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 17:35
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : stereo_matcher.py


import cv2
from utils import *
__methods_name__ = ["BM", "SGBM"]


class StereoMatcher(object):

    def __init__(self, methods, is_rectified=True, camera_params_file=None, **matcherKwargs):
        """
        :param methods: stereo match methods name, currently support BM and SGBM.
        :param is_rectified: if the image pair is rectified.
        :param camera_params_file: if isn't rectified, camera parameters must be need to rectify stereo image pair.
        :param matcherKwargs: parameters to initialize matcher.
                              "BM": numDisparities(the disparity search range), blockSize
                              "SGBM": minDisparity, numDisparities, blockSize, P1, P2,
                                      disp12MaxDiff, preFilterCap, uniquenessRatio,
                                      speckleWindowSize, speckleRange, mode
        """
        assert is_rectified or camera_params_file is not None, \
            "images must be rectified or it has camera_params file."
        assert methods in __methods_name__, "not support {}, only support {}.".format(methods, __methods_name__)
        self.methods = methods
        self.is_gray = True
        if methods == "SGBM":
            self.matcher = cv2.StereoSGBM_create(**matcherKwargs)
        else:
            self.matcher = cv2.StereoBM_create(**matcherKwargs)
        self.is_rectified = is_rectified
        if not is_rectified and camera_params_file is not None:
            self.camera_params = json_load(camera_params_file)
        self.matcherKwargs = matcherKwargs
        print(self.__str__())

    def predict(self, imgL, imgR):
        if self.is_gray:
            imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY) if len(imgL.shape) > 2 else imgL
            imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY) if len(imgR.shape) > 2 else imgR
        if not self.is_rectified:
            imgL, imgR = rectifyImagePair(imgL, imgR, cameraParams=self.camera_params)
        return self.matcher.compute(imgL, imgR)

    def __str__(self):
        obj_str  = "create Stereo Matcher {} for ".format(self.methods)
        obj_str += "rectified image pair, " if self.is_rectified else "not rectified image pair, "
        obj_str += "matcher parameters: "
        for k, v in self.matcherKwargs.items():
            obj_str += "{}:{}, ".format(k, v)
        return obj_str


if __name__ == "__main__":
    imgL = cv2.imread("./data/stereo_pairs/tsukuba_l.png", flags=0)
    imgR = cv2.imread("./data/stereo_pairs/tsukuba_r.png", flags=0)
    matcher = StereoMatcher(methods="BM", is_rectified=True, numDisparities=16, blockSize=15)
    disparity = matcher.predict(imgL, imgR)
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow('disparity', disparity)
    cv2.waitKey()
