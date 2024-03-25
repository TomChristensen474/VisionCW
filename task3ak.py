import os
from pathlib import Path
from typing import List, Tuple
import cv2 as cv
from dataclasses import dataclass
import numpy as np

def task3(folderName: str) -> float:
    # TODO: don't harcode paths
    path_dir = os.getcwd() + "/IconDataset/png/01-lighthouse.png"

    img = cv.imread(str(path_dir))

    get_blobs(img=img)
 
    return 0.0

def get_blobs(img):
    GAUSSIAN_SIZE = 3
    GAUSSIAN_SD = 0
    img_gaussian = cv.GaussianBlur(img, (GAUSSIAN_SIZE, GAUSSIAN_SIZE), GAUSSIAN_SD)
    img_gray = cv.cvtColor(img_gaussian, cv.COLOR_BGR2GRAY)
    
    kernel_size = 3
    img_laplacian = cv.Laplacian(img_gray, cv.CV_16S, ksize=kernel_size)
    img_output = cv.convertScaleAbs(img_laplacian)
    cv.imshow("", img_output)
    cv.waitKey(0)
    return
    # filter_scales = []
    # for filter_scale in filter_scales:
    #     pass
    # # 1. 
    # pass

def get_descriptor():
    pass


def get_feature():
    pass


if __name__ == "__main__":
    task3("Task3Dataset")
