import cv2
import numpy as np
from scipy.interpolate import interp1d


def create_mask_by_hull(input_image):
    '''
    input_image - black objects on white background
    '''

    _, input_image_thresholded = cv2.threshold(input_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(255 - input_image_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    combined = np.concatenate((contours[0], contours[1]), axis=0)
    for contour in contours[2:]:
        combined = np.concatenate((combined, contour), axis=0)

    hull = cv2.convexHull(combined)

    texture_mask = np.zeros_like(input_image).astype(np.float32)
    texture_mask = cv2.fillPoly(texture_mask, [hull], 1)

    return texture_mask


def create_mask_by_shape(input_image):
    _, input_image_thresholded = cv2.threshold(input_image.astype(np.uint8), 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(255 - input_image_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    texture_mask = np.zeros_like(input_image).astype(np.float32)
    texture_mask = cv2.fillPoly(texture_mask, [max_contour], 1)

    return texture_mask


def merge_segmentation_mask(mask):
    """
    Merge every channel of segmentation mask into single channel mask with values [0, 1]
    """
    mask = mask // 255
    merged_mask = np.zeros([mask.shape[0], mask.shape[1]])

    for x in range(mask.shape[2]):
        merged_mask[mask[:, :, x] > 0] = 1

    return merged_mask


def get_edges_positions(image):
    '''
    Get positions of most Top, Bottom, Left, Right pixels of black object on white canvas
    '''
    # revert colors
    image2 = 255-image

    _, image3 = cv2.threshold(image2, 64, 255, cv2.THRESH_BINARY)

    positions = np.nonzero(image3)

    return [positions[0].min(), positions[0].max(), positions[1].min(), positions[1].max()]


def crop_to_object(image, segmentation_mask=None):
    top, bottom, left, right = get_edges_positions(image)

    image = image[top:bottom, left:right]

    if segmentation_mask is not None:
        segmentation_mask = segmentation_mask[top:bottom, left:right]
        return image, segmentation_mask

    return image


def coordinates_smoothing(x_coordinates, y_coordinates, additional_canvas_size):
    t = np.arange(len(x_coordinates))
    ti = np.linspace(0, t.max(), 10 * t.size)
    xi = interp1d(t, x_coordinates, kind='cubic')(ti)
    yi = interp1d(t, y_coordinates, kind='cubic')(ti)

    xi = xi.astype(int)
    yi = yi.astype(int)

    x_min = min(xi)
    x_max = max(xi)
    y_min = min(yi)
    y_max = max(yi)

    new_x_shape = x_max - x_min
    new_y_shape = y_max - y_min

    x_movement = abs(x_min) + additional_canvas_size
    y_movement = abs(y_min) + additional_canvas_size

    xi = [coord + x_movement for coord in xi]
    yi = [coord + y_movement for coord in yi]

    return xi, yi, x_movement, y_movement, new_x_shape, new_y_shape
