import argparse
import cv2
import numpy as np
import pandas as pd
import os
import math
from matplotlib import pyplot as plt


def gaussian(img):
    return cv2.GaussianBlur(img, (15, 15), 0)


def canny(img):
    return cv2.Canny(img, 100, 200, None, 3)


def get_hough_lines(img):
    return cv2.HoughLines(img, 1, np.pi / 180, 75)


def draw_hough_lines(img, lines):
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(img, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
    return img, len(lines)


def get_angles_in_image(img, debug_mode=False):

    # cv2.imshow("Original", img)
    # kernel = np.ones((5, 5), np.uint8)
    # cv2.erode(img, kernel, iterations=1)
    # if debug_mode:
    #     cv2.imshow("Original", img)
    #     cv2.imshow("Eroded", img)
    #     cv2.waitKey(0)

    gaussian = cv2.GaussianBlur(img, (7, 7), 0)
    # if debug_mode:
    #     cv2.imshow("Gaussian", gaussian)
    #     cv2.waitKey(0)

    edges = cv2.Canny(gaussian, 100, 200, None, 5)
    if debug_mode:
        cv2.imshow("Canny", edges)
        cv2.waitKey(0)

    cdst = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 75)

    thetas = []

    assert lines is not None, "No lines detected"

    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        thetas.append(math.degrees(theta))
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    if debug_mode:
        cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        cv2.imshow("Original", img)
        cv2.waitKey(0)

    thetas.sort()
    # print(thetas)
    for theta in thetas:
        removed_thetas = []
        thetas_to_average = []
        removing = False
        for t in thetas:
            if abs(theta - t) < 5:
                removed_thetas.append(t)
                thetas_to_average.append(t)
                # thetas.remove(t)
                removing = True

            elif abs(theta - (t - 180)) < 5:
                removed_thetas.append(t)
                thetas_to_average.append(t - 180)
                removing = True

        if removing:
            mean_thetas = sum(thetas_to_average) / len(thetas_to_average)
            if mean_thetas < 0:
                mean_thetas += 180
            thetas.append(mean_thetas)
            for t in removed_thetas:
                thetas.remove(t)

    gradients = []
    for theta in thetas:
        gradients.append(-math.cos(math.radians(theta))/math.sin(math.radians(theta)))

    # print(thetas)
    # print(thetas)
    assert len(thetas) == 2, "Error: More or less than 2 lines calculated"
    # angle = 180 - (thetas[0] + (180 - thetas[1]))
    # print("PRODUCT = ", gradients[0]*gradients[1])
    print(gradients)
    if (gradients[0] * gradients[1]) < 0:
        angle = 180 - (thetas[1] - thetas[0])
    else:
        angle = thetas[1] - thetas[0]

    # print(thetas[0], thetas[1])
    print(gradients[0]*gradients[1], " -- ", abs(angle))

    return abs(angle)


def testTask1(folderName):
    # assume that this folder name has a file list.txt that contains the annotation

    # Write code to read in each image
    # Write code to process the image
    # Write your code to calculate the angle and obtain the result as a list predAngles
    # Calculate and provide the error in predicting the angle for each image
    task1Data = pd.read_csv(folderName + "/list.txt")

    img = cv2.imread(os.path.join(folderName, "image4.png"))
    assert img is not None, "file could not be read, check with os.path.exists()"

    print(get_angles_in_image(img, True))

    # angles = []
    # for files in os.listdir(folderName):
    #     if files.endswith(".png"):
    #         img = cv2.imread(os.path.join(folderName, files))
    #         assert img is not None, "file could not be read, check with os.path.exists()"

    #         angles.append(get_angles_in_image(img))

    # print(task1Data, angles)
    # return angles


def testTask2(iconDir, testDir):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    return (Acc, TPR, FPR, FNR)


def testTask3(iconFolderName, testFolderName):
    # assume that test folder name has a directory annotations with a list of csv files
    # load train images from iconDir and for each image from testDir, match it with each class from the iconDir to find the best match
    # For each predicted class, check accuracy with the annotations
    # Check and calculate the Intersection Over Union (IoU) score
    # based on the IoU determine accuracy, TruePositives, FalsePositives, FalseNegatives
    return (Acc, TPR, FPR, FNR)


if __name__ == "__main__":

    # parsing the command line path to directories and invoking the test scripts for each task
    parser = argparse.ArgumentParser("Data Parser")
    parser.add_argument(
        "--Task1Dataset",
        help="Provide a folder that contains the Task 1 Dataset.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--IconDataset",
        help="Provide a folder that contains the Icon Dataset for Task2 and Task3.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--Task2Dataset",
        help="Provide a folder that contains the Task 2 test Dataset.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--Task3Dataset",
        help="Provide a folder that contains the Task 3 test Dataset.",
        type=str,
        required=False,
    )
    args = parser.parse_args()
    if args.Task1Dataset != None:
        # This dataset has a list of png files and a txt file that has annotations of filenames and angle
        testTask1(args.Task1Dataset)
    if args.IconDataset != None and args.Task2Dataset != None:
        # The Icon dataset has a directory that contains the icon image for each file
        # The Task2 dataset directory has two directories, an annotation directory that contains the annotation and a png directory with list of images
        testTask2(args.IconDataset, args.Task2Dataset)
    if args.IconDataset != None and args.Task3Dataset != None:
        # The Icon dataset directory contains an icon image for each file
        # The Task3 dataset has two directories, an annotation directory that contains the annotation and a png directory with list of images
        testTask3(args.IconDataset, args.Task3Dataset)
