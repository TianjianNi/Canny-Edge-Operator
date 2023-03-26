import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


# The function displays the input image using the OpenCV library
def display_image(image, image_title):

    image = image.astype(np.uint8)
    cv2.imshow(image_title, image)
    cv2.waitKey(0)  # wait for a keyboard input
    cv2.destroyAllWindows()


# The function returns 7 x 7 Gaussian Mask
def gaussian_mask_seven_by_seven():

    row0 = np.array([1, 1, 2, 2, 2, 1, 1])
    row1 = np.array([1, 2, 2, 4, 2, 2, 1])
    row2 = np.array([2, 2, 4, 8, 4, 2, 2])
    row3 = np.array([2, 4, 8, 16, 8, 4, 2])
    row4 = row2
    row5 = row1
    row6 = row0
    return np.vstack((row0, row1, row2, row3, row4, row5, row6))


# The function returns the stack of the 4 masks which will be used to compute the gradients at 0, 45, 90, 135 degrees
def masks():

    g0 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    g1 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    g2 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    g3 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])
    stack = np.stack((g0, g1, g2, g3), axis=0)
    return stack


# The function takes one input image and mask and compute convolution of the two
def convolution(image, mask):

    # size of the input image
    image_size_i = int(image.shape[0])
    image_size_j = int(image.shape[1])

    # size of the mask
    mask_size_i = int(mask.shape[0])
    mask_size_j = int(mask.shape[1])

    # sizes of the output image
    output_size_i = image_size_i - mask_size_i + 1
    output_size_j = image_size_j - mask_size_j + 1
    output = np.zeros((output_size_i, output_size_j))

    # compute each output image's pixel through computing convolution
    for i in range(image_size_i):
        for j in range(image_size_j):
            if i > image_size_i - mask_size_i or j > image_size_j - mask_size_j:
                continue
            else:
                output[i, j] = (mask * image[i: i + mask_size_i, j: j + mask_size_j]).sum()

    return output


class CannyEdgeDetector:

    def __init__(self,
                 image_name=None):

        self.img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        # Output after gaussian smoothing and normalization
        self.gaussian_smoothed = self.gaussian_smoothing()
        # Output after compute convolution between the gaussian smoothed image and each of the four masks
        self.gradients = self.gradient_operation()
        # Output of the edge magnitudes after taking the maximum of the absolute values of the responses
        # from the four masks, and then divide by 4
        self.magnitudes = self.calculate_magnitudes()
        # Output of the gradient angles after selecting the index of the mask that produces the maximum response
        self.quantized_angles = self.quantize_angles()
        # Output after non-maxima suppression
        self.output_after_non_maxima_suppression = self.non_maxima_suppression()
        # Output of the quantiles after thresholding
        self.edge_map_at_25, self.edge_map_at_50, self.edge_map_at_75 = self.thresholding()

    def get_original_image(self):
        return self.img

    # Gaussian smoothing step in Canny Edge Detector
    def gaussian_smoothing(self):

        gaussian_mask = gaussian_mask_seven_by_seven()
        # Use the 7 x 7 Gaussian mask for smoothing the input image
        output_after_gaussian_smoothing = convolution(self.get_original_image(), gaussian_mask)
        # perform normalization by dividing by the sum of the entries for the given mask at each pixel location
        output_after_normalization = output_after_gaussian_smoothing / 140
        return output_after_normalization

    # Get the output after the Gaussian smoothing step
    def get_gaussian_smoothed(self):
        return self.gaussian_smoothed

    # Gradient operation step to compute gradients
    # Compute convolution between the gaussian smoothed image and each of the four masks
    def gradient_operation(self):

        masks_for_gradient_operation = masks()
        gaussian_smoothed = self.get_gaussian_smoothed()
        # Compute convolution of the Gaussian smoothed image with gradient mask at 0 degree
        h0 = abs(convolution(gaussian_smoothed, masks_for_gradient_operation[0]))
        # Compute convolution of the Gaussian smoothed image with gradient mask at 45 degree
        h1 = abs(convolution(gaussian_smoothed, masks_for_gradient_operation[1]))
        # Compute convolution of the Gaussian smoothed image with gradient mask at 90 degree
        h2 = abs(convolution(gaussian_smoothed, masks_for_gradient_operation[2]))
        # Compute convolution of the Gaussian smoothed image with gradient mask at 135 degree
        h3 = abs(convolution(gaussian_smoothed, masks_for_gradient_operation[3]))
        stack = np.stack((h0, h1, h2, h3), axis=0)
        return stack

    # Get the output after compute convolution between the gaussian smoothed image and each of the four masks
    def get_gradients(self):
        return self.gradients

    # Gradient operation step to compute edge magnitudes
    def calculate_magnitudes(self):

        gradients = self.get_gradients()
        # Calculate the maximum of the absolute values of the gradients/responses generated from the four masks
        edge_magnitude = np.max(gradients, axis=0)
        # Edge magnitude is normalized through dividing by 4
        edge_magnitude_after_normalization = edge_magnitude / 4
        return edge_magnitude_after_normalization

    # Get the edge magnitudes
    def get_magnitudes(self):
        return self.magnitudes

    # Gradient operation step to compute gradient angles
    def quantize_angles(self):

        gradients = self.get_gradients()
        # the quantized angle equals to the index of the mask that produces the maximum response
        output_quantized_angles = np.argmax(gradients, axis=0)
        return output_quantized_angles

    def get_angles(self):
        return self.quantized_angles

    # Non-maxima suppression step
    def non_maxima_suppression(self):

        quantized_angles = self.get_angles()
        size_i, size_j = quantized_angles.shape
        output_non_maxima_suppression = np.zeros((size_i, size_j))
        magnitudes = self.get_magnitudes()
        # Suppress values along the line of gradient that are not peak values of a ridge
        for row in range(1, size_i - 1):
            for col in range(1, size_j - 1):
                local_magnitude = magnitudes[row][col]
                if quantized_angles[row][col] == 0:
                    if magnitudes[row][col + 1] < local_magnitude \
                            and magnitudes[row][col - 1] < local_magnitude:
                        output_non_maxima_suppression[row][col] = local_magnitude
                elif quantized_angles[row][col] == 1:
                    if magnitudes[row - 1][col + 1] < local_magnitude \
                            and magnitudes[row + 1][col - 1] < local_magnitude:
                        output_non_maxima_suppression[row][col] = local_magnitude
                elif quantized_angles[row][col] == 2:
                    if magnitudes[row - 1][col] < local_magnitude \
                            and magnitudes[row + 1][col] < local_magnitude:
                        output_non_maxima_suppression[row][col] = local_magnitude
                else:
                    if quantized_angles[row - 1][col - 1] < local_magnitude \
                            and magnitudes[row + 1][col + 1] < local_magnitude:
                        output_non_maxima_suppression[row][col] = local_magnitude
        return output_non_maxima_suppression

    # Get the output after non-maxima suppression
    def get_non_maxima_suppression_result(self):
        return self.output_after_non_maxima_suppression

    # Thresholding step
    def thresholding(self):

        output_non_maxima_suppression = self.get_non_maxima_suppression_result()
        # exclude pixels with zero gradient magnitude when determining the percentiles
        output_non_maxima_suppression[output_non_maxima_suppression == 0] = np.nan

        # thresholds chosen at the 25th, 50th and 75th percentiles after non-maxima suppression
        quantile25 = np.nanquantile(output_non_maxima_suppression, 0.25)
        quantile50 = np.nanquantile(output_non_maxima_suppression, 0.50)
        quantile75 = np.nanquantile(output_non_maxima_suppression, 0.75)

        # Binary edge maps at the 25th, 50th, and 75th percentiles
        output_non_maxima_suppression = self.get_non_maxima_suppression_result()
        edge_map_at_25 = np.where(output_non_maxima_suppression > quantile25, 255, 0)
        edge_map_at_50 = np.where(output_non_maxima_suppression > quantile50, 255, 0)
        edge_map_at_75 = np.where(output_non_maxima_suppression > quantile75, 255, 0)

        return edge_map_at_25, edge_map_at_50, edge_map_at_75

    # Get the binary edge maps at the 25th, 50th, and 75th percentiles
    def get_edge_maps(self):
        return self.edge_map_at_25, self.edge_map_at_50, self.edge_map_at_75

    def display_original_image(self):
        size_i, size_j = self.img.shape
        output = self.img
        title = "Original image " + str(size_i) + " x " + str(size_j)
        display_image(output, title)

    # Output image (1): image result after Gaussian smoothing
    def display_image_after_gaussian_smoothing(self):

        # The output images should be of the same size as the original input image
        # after replacing undefined values with 0’s
        size_i, size_j = self.img.shape
        output = np.zeros((size_i, size_j))
        output[3: size_i - 3, 3: size_j - 3] = self.get_gaussian_smoothed()

        size_i, size_j = output.shape
        title = "Output image (1): image result after Gaussian smoothing " + str(size_i) + " x " + str(size_j)
        display_image(output, title)
        cv2.imwrite(os.path.join(os.path.expanduser('~'), 'Desktop', 'gaussian smoothing.bmp'), output)

    # Output image (2): image result after normalized magnitude
    def display_image_after_normalized_magnitude(self):

        # The output images should be of the same size as the original input image
        # after replacing undefined values with 0’s
        size_i, size_j = self.img.shape
        output = np.zeros((size_i, size_j))
        output[4: size_i - 4, 4: size_j - 4] = self.get_magnitudes()

        size_i, size_j = output.shape
        title = "Output image (2): image result after normalized magnitude " + str(size_i) + " x " + str(size_j)
        display_image(self.get_magnitudes(), title)
        cv2.imwrite(os.path.join(os.path.expanduser('~'), 'Desktop', 'normalized magnitude.bmp'), output)

    # Output image (3): image result after non-maxima suppression
    def display_image_after_non_maxima_suppression(self):

        # The output images should be of the same size as the original input image
        # after replacing undefined values with 0’s
        size_i, size_j = self.img.shape
        output = np.zeros((size_i, size_j))
        output[4: size_i - 4, 4: size_j - 4] = self.get_non_maxima_suppression_result()

        size_i, size_j = output.shape
        title = "Output image (3): image result after non-maxima suppression " + str(size_i) + " x " + str(size_j)
        display_image(self.get_non_maxima_suppression_result(), title)
        cv2.imwrite(os.path.join(os.path.expanduser('~'), 'Desktop', 'non-maxima suppression.bmp'), output)

    # Output image (4): binary edge maps for thresholds chosen at the 25th, 50th and 75th percentiles
    def display_images_at_different_percentiles(self):

        edge_map_at_25, edge_map_at_50, edge_map_at_75 = self.get_edge_maps()

        # The output images should be of the same size as the original input image
        # after replacing undefined values with 0’s
        size_i, size_j = self.img.shape
        output = np.zeros((size_i, size_j))
        output[4: size_i - 4, 4: size_j - 4] = edge_map_at_25

        size_i, size_j = output.shape
        title25 = "Output image (4): 25th percentiles " + str(size_i) + " x " + str(size_j)
        display_image(output, title25)
        cv2.imwrite(os.path.join(os.path.expanduser('~'), 'Desktop', '25th percentiles.bmp'), output)

        # The output images should be of the same size as the original input image
        # after replacing undefined values with 0’s
        size_i, size_j = self.img.shape
        output = np.zeros((size_i, size_j))
        output[4: size_i - 4, 4: size_j - 4] = edge_map_at_50

        size_i, size_j = output.shape
        title50 = "Output image (4): 50th percentiles " + str(size_i) + " x " + str(size_j)
        display_image(output, title50)
        cv2.imwrite(os.path.join(os.path.expanduser('~'), 'Desktop', '50th percentiles.bmp'), output)

        # The output images should be of the same size as the original input image
        # after replacing undefined values with 0’s
        size_i, size_j = self.img.shape
        output = np.zeros((size_i, size_j))
        output[4: size_i - 4, 4: size_j - 4] = edge_map_at_75

        size_i, size_j = output.shape
        title75 = "Output image (4): 75th percentiles " + str(size_i) + " x " + str(size_j)
        display_image(output, title75)
        cv2.imwrite(os.path.join(os.path.expanduser('~'), 'Desktop', '75th percentiles.bmp'), output)

    # Output image (5): a histogram of the normalized magnitude image after non-maxima suppression
    def display_histogram_of_the_normalized_magnitude(self):

        plt.title("Histogram of the normalized magnitude after non-maxima suppression")
        plt.hist(self.get_non_maxima_suppression_result().ravel(), 256, [0, 256]);
        plt.show()

    def out_all_images(self):

        self.display_original_image()
        self.display_image_after_gaussian_smoothing()
        self.display_image_after_normalized_magnitude()
        self.display_image_after_non_maxima_suppression()
        self.display_images_at_different_percentiles()
        self.display_histogram_of_the_normalized_magnitude()


if __name__ == "__main__":

    test = CannyEdgeDetector("Peppers.bmp")
    test.out_all_images()