
import cv2
import numpy as np

# Citations
# Thinning
# https://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm#:~:text=Thinning%20is%20a%20morphological%20operation,is%20particularly%20useful%20for%20skeletonization.

# Hit and miss
# https://docs.opencv.org/4.x/db/d06/tutorial_hitOrMiss.html
# https://docs.opencv.org/3.4/db/d06/tutorial_hitOrMiss.html

class Thinner:
    def __init__(self):
        # J's
        self.structuring_elements = [
            np.array([[0, 0, 0],
                      [0, 1, 0],
                      [1, 1, 1]], dtype=np.uint8),
            np.array([[0, 0, 0],
                      [1, 1, 0],
                      [0, 1, 0]], dtype=np.uint8)
        ]

    # input: binary image
    # output: thinned binary image
    # thinning of an image I by a structuring element J is:
    # thin(I,J) = I - hit-and-miss(I,J) 
    def thin_image(self, image_binary):
        I = image_binary.copy()
        while True:
            prev_I = I.copy()
            for J in self.structuring_elements:
                hit_and_miss_I_J = cv2.morphologyEx(I,cv2.MORPH_HITMISS,J)
                thin_I_J = cv2.subtract(I,hit_and_miss_I_J)
            if np.array_equal(prev_I,thin_I_J):
                return thin_I_J
            I = thin_I_J


if __name__ == "__main__":
    image = cv2.imread('./Task1Dataset/image4.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)

    thinner = Thinner()
    thinned_img = thinner.thin_image(binary_image)

    # cv2.imwrite("binary_image.png", binary_image)
    # cv2.imwrite("thinned_image.png", thinned_img)


    