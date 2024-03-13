import argparse
import cv2
import numpy as np
from natsort import natsorted
import pandas as pd
import os
from angle_calculator import get_angles_in_image


def testTask1(folderName):
    # assume that this folder name has a file list.txt that contains the annotation
    # Write code to read in each image
    # Write code to process the image
    # Write your code to calculate the angle and obtain the result as a list predAngles
    # Calculate and provide the error in predicting the angle for each image
    task1Data = pd.read_csv(folderName + "/list.txt")

    img = cv2.imread(os.path.join(folderName, "image5.png"))
    assert img is not None, "file could not be read, check with os.path.exists()"

    # print(get_angles_in_image(img, True))

    angles = []

    #natsorted in order to get process files in correct order
    for file in natsorted(os.listdir(folderName)):
        print(file)
        if file.endswith(".png"):
            img = cv2.imread(os.path.join(folderName, file))
            assert img is not None, "file could not be read, check with os.path.exists()"

            angles.append(get_angles_in_image(img, False))

    task1Data.insert(2, "predAngles", angles, True)

    # Calculate and provide the error in predicting the angle for each image
    task1Data["error"] = abs(task1Data["AngleInDegrees"] - task1Data["predAngles"])
    print(task1Data)

    # return (Acc, TPR, FPR, FNR)


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
