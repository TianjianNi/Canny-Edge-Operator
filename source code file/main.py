import numpy as np

import model

if __name__ == "__main__":

    test = model.CannyEdgeDetector("Barbara.bmp")  # Replace with the name of the image that you want to test
    test.out_all_images()


