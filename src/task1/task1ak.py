from pathlib import Path
from dataclasses import dataclass
from typing import Iterator
import numpy as np
import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def task1(folderName: str):
    folderPath = Path("/Users/aaronkher/Documents/VScodeProjects/VisionCW") / folderName
    imagePath = folderPath / "image1.png"
    image = cv.imread(str(imagePath), cv.IMREAD_GRAYSCALE)


    print(image)


    
    # plt.figure(figsize=(8, 6))
    # plt.imshow(edges, cmap='gray')
    # plt.title('canny')
    # plt.show()

if __name__ == "__main__":
    task1("Task1Dataset")
