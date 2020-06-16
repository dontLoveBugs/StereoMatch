#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 22:29
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
# @File    : calibrator.py


import cv2
import glob
from utils import *
import numpy as np


def findCornerMultiScale(image, boardSize, scales=[1]):
    assert isinstance(scales, list) and len(scales) >= 1, "scales must be list and length >= 1"
    found = False
    corners = None
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for scale in scales:
        # print(scale)
        if scale == 1:
            timg = image.copy()
        else:
            timg = cv2.resize(image, None, fx=scale, fy=scale)
        # print("timg shape:", timg.shape)

        found, corners = cv2.findChessboardCorners(timg, boardSize, None,
                                                   cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            continue
        corners = corners * 1. / scale
    if not found:
        return None
    corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
    return corners


class Calibrator(object):

    def __init__(self, images_dir, border_size):
        # assert camera_params_file is None or isinstance(camera_params_file, str), \
        #     "camera params file is None and its type is not str."
        assert isinstance(images_dir, str), \
            "image dir is None and its type is not str."
        assert isinstance(border_size, tuple) and len(border_size) == 2, \
            "border size should be a tuple and its length should be 2."
        # self.camera_params_file = camera_params_file
        self.images_dir = images_dir
        self.border_size = border_size
        self.stereo_camera_params = None

    def calibrate(self):
        left_images, right_images = [], []
        for name in glob.glob(self.images_dir + "/left*"):
            left_images.append(name)
        for name in glob.glob(self.images_dir + "/right*"):
            right_images.append(name)

        assert len(left_images) == len(right_images), "Number of the left images must be same with the right ones."
        left_images.sort()
        right_images.sort()

        # keypoints params
        scales = [1, 2]
        square_size = 20.  # Set this to your actual square size

        # find corners
        print("finding corners...")
        good_image_pairs = []
        image_points = [[], []]
        image_shape = None
        for i in range(len(left_images)):
            left_image = cv2.imread(left_images[i], 0)
            if image_shape is None:
                image_shape = left_image.shape
            else:
                if image_shape != left_image.shape:
                    print("the shape of left image {} is {}, not same with {}.".format(
                        i, left_image.shape, image_shape))
                    continue
            left_corners = findCornerMultiScale(left_image, self.border_size, scales)
            if left_corners is None: continue

            right_image = cv2.imread(right_images[i], 0)
            if image_shape is None:
                image_shape = right_image.shape
            else:
                if image_shape != right_image.shape:
                    print("the shape of left image {} is {}, not same with {}.".format(
                        i, right_image.shape, image_shape))
                    continue

            right_corners = findCornerMultiScale(right_image, self.border_size, scales)
            if right_corners is None: continue

            good_image_pairs.append([left_image, right_image])
            image_points[0].append(left_corners)
            image_points[1].append(right_corners)

        n_image_pairs = len(good_image_pairs)
        assert n_image_pairs > 1, "Error: number of image pairs is {}, " \
                                  "which is too little pairs to run the calibration".format(n_image_pairs)

        print("find {} good image pairs.".format(n_image_pairs))
        objp = np.zeros((np.prod(self.border_size), 3), np.float32)
        objp[:, :2] = np.indices(self.border_size).T.reshape(-1, 2)
        objp *= square_size
        object_points = [objp] * n_image_pairs

        # initialize camera matrix
        h, w = image_shape
        camera_matrix = list()
        camera_matrix.append(cv2.initCameraMatrix2D(object_points, image_points[0], (w, h), 0))
        camera_matrix.append(cv2.initCameraMatrix2D(object_points, image_points[1], (w, h), 0))

        # calibration
        print("calibrating...")
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
        ret, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = \
            cv2.stereoCalibrate(object_points, image_points[0], image_points[1], camera_matrix[0],
                                None, camera_matrix[1], None, (w, h),
                                flags=cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_USE_INTRINSIC_GUESS |
                                      cv2.CALIB_SAME_FOCAL_LENGTH | cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3 |
                                      cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5,
                                criteria=term)

        # save calibration params
        # print("cam1 {}, dist1 {}".format(cameraMatrix1, distCoeffs1))
        # print('R={}, T={}, E={}, F={}.'.format(R, T, E, F))
        self.stereo_camera_params = dict(cameraMatrix1=cameraMatrix1, distCoeffs1=distCoeffs1,
                                         cameraMatrix2=cameraMatrix2, distCoeffs2=distCoeffs2,
                                         R=R, T=T, E=E, F=F)

    def save(self, filename):
        # print(self.stereo_camera_params)
        json_save(filename, self.stereo_camera_params)
        print("save stereo cameras paramsters in {}.".format(filename))


if __name__ == "__main__":
    calibrator = Calibrator(images_dir="./data/calibration_images", border_size=(9, 6))
    calibrator.calibrate()
    calibrator.save("stereo_camera.json")

    left_image = cv2.imread("data/calibration_images/left01.jpg", flags=0)
    right_image = cv2.imread("data/calibration_images/right01.jpg", flags=0)

    cv2.imshow("left image", left_image)
    cv2.waitKey()

    cameraParams = json_load("stereo_camera.json")

    print(left_image.shape)
    h, w = left_image.shape
    rectifiedLeftImage, rectifiedRightImage = rectifyImagePair(left_image, right_image, (h, w), cameraParams)

    print("rectifiedLeftImage shape:", rectifiedLeftImage.shape)
    result = np.concatenate((rectifiedLeftImage, rectifiedRightImage), axis=1)
    result[::20, :] = 0
    cv2.imshow("rectifiedImages", result)
    cv2.waitKey()

    # image = cv2.imread("./data/right01.jpg", 0)
    # print(image.shape)
    # cv2.imshow("image", image)
    # cv2.waitKey()
    #
    # corners = findCornerMultiScale(image, boardSize=(7,6), scales=[1, 2])
    # print(corners.shape)
    # image = cv2.drawChessboardCorners(image, (7,6), corners, corners is not None)
    # cv2.imshow("image", image)
    # cv2.waitKey()
