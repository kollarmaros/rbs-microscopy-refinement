import random

import cv2
import numpy as np
from scipy import ndimage


def create_central_pair(settings, fill):
    # create canvas for central pair
    canvas_dimension = (settings['canvas_shape'], settings['canvas_shape'])
    canvas = np.uint8(np.ones(canvas_dimension) * 255)

    color = settings["color"]  # set color

    if settings["defect"] == "multiple":
        number_of_tubules = random.choice([3, 4])
        coordinates_axes = []

        first_row_divider = 2
        second_row_multipier = 1.4
        if number_of_tubules == 3:
            three_tubules_width_max_multiplier = 2
        else:
            three_tubules_width_max_multiplier = 1

        # First
        coordinates = (random.randrange(settings["first_start_width_range"][0],
                                        settings["first_start_width_range"][1]),
                       random.randrange(settings["first_start_height_range"][0],
                                        settings["first_start_height_range"][1]) // first_row_divider)  # w, h
        axesLength = (random.randrange(settings["axes_length_range"][0], settings["axes_length_range"][1]),
                      random.randrange(settings["axes_length_range"][0], settings["axes_length_range"][1]))
        coordinates_axes.append([coordinates, axesLength])

        # Second
        tubule_gap = random.randrange(settings["tubule_gap_range"][0], settings["tubule_gap_range"][1])
        axesLength = (random.randrange(settings["axes_length_range"][0], settings["axes_length_range"][1]),
                      random.randrange(settings["axes_length_range"][0], settings["axes_length_range"][1]))
        coordinates_axes.append([(coordinates[0] + 2 * axesLength[0] + tubule_gap, coordinates[1]), axesLength])

        # third
        coordinates = (random.randrange(settings["first_start_width_range"][0],
                                        settings["first_start_width_range"][1] * three_tubules_width_max_multiplier),
                       int(random.randrange(settings["first_start_height_range"][0],
                                            settings["first_start_height_range"][1]) * second_row_multipier))  # w, h
        axesLength = (random.randrange(settings["axes_length_range"][0], settings["axes_length_range"][1]),
                      random.randrange(settings["axes_length_range"][0], settings["axes_length_range"][1]))
        coordinates_axes.append([coordinates, axesLength])

        if number_of_tubules == 4:
            # fourth
            tubule_gap = random.randrange(settings["tubule_gap_range"][0], settings["tubule_gap_range"][1])
            axesLength = (random.randrange(settings["axes_length_range"][0], settings["axes_length_range"][1]),
                          random.randrange(settings["axes_length_range"][0], settings["axes_length_range"][1]))
            coordinates_axes.append([(coordinates[0] + 2 * axesLength[0] + tubule_gap, coordinates[1]), axesLength])

    else:  # No defect or single tubule
        coordinates_axes = []
        coordinates = (random.randrange(settings["first_start_width_range"][0],
                                        settings["first_start_width_range"][1]),
                       random.randrange(settings["first_start_height_range"][0],
                                        settings["first_start_height_range"][1]))  # w, h
        axesLength = (random.randrange(settings["axes_length_range"][0], settings["axes_length_range"][1]),
                      random.randrange(settings["axes_length_range"][0], settings["axes_length_range"][1]))
        coordinates_axes.append([coordinates, axesLength])
        if settings["defect"] != "single_tubule":
            tubule_gap = random.randrange(settings["tubule_gap_range"][0], settings["tubule_gap_range"][1])
            axesLength = (random.randrange(settings["axes_length_range"][0], settings["axes_length_range"][1]),
                          random.randrange(settings["axes_length_range"][0], settings["axes_length_range"][1]))
            coordinates_axes.append([(coordinates[0] + 2 * axesLength[0] + tubule_gap, coordinates[1]), axesLength])

    # Draw tubules
    for center_coordinates, axesLength in coordinates_axes:
        angle = random.randrange(settings["tubule_angle_range"][0], settings["tubule_angle_range"][1])
        thickness = random.randrange(settings["thickness_range"][0], settings["thickness_range"][1])

        # Draw tubule
        canvas = cv2.ellipse(canvas, center_coordinates, axesLength,
                             angle, 0, 360, color, thickness)
        # Fill tubule
        if fill == "yes" or (fill == "partial" and bool(random.getrandbits(1))):
            cv2.ellipse(canvas, center_coordinates, axesLength,
                        angle, 0, 360, color, -1)

    # rotation
    rotation_angle = random.randrange(settings["rotation_angle_range"][0], settings["rotation_angle_range"][1])
    canvas = np.uint8(np.clip(ndimage.rotate(canvas, rotation_angle, cval=255), 0, 255))

    return canvas


def create_outer_pair(settings, fill, single_tubule=False):
    # create canvas for central pair
    canvas_dimension = (settings["canvas_shape"], settings["canvas_shape"])
    canvas = np.uint8(np.ones(canvas_dimension) * 255)

    color = settings["color"]  # set color

    ##############
    # First tubule
    ##############
    center_coordinates = (random.randrange(settings["first_start_width_range"][0],
                                           settings["first_start_width_range"][1]),
                          random.randrange(settings["first_start_height_range"][0],
                                           settings["first_start_height_range"][1]))  # w, h
    axesLength = (random.randrange(settings["axes_length_range"][0], settings["axes_length_range"][1]),
                  random.randrange(settings["axes_length_range"][0], settings["axes_length_range"][1]))  # w, h
    angle = random.randrange(settings["tubule_angle_range"][0], settings["tubule_angle_range"][1])
    thickness = random.randrange(settings["thickness_range"][0], settings["thickness_range"][1])

    # Draw tubule
    canvas = cv2.ellipse(canvas, center_coordinates, axesLength,
                         angle, 0, 360, color, thickness)
    # Fill tubule
    if fill == "yes" or (fill == "partial" and bool(random.getrandbits(1))):
        cv2.ellipse(canvas, center_coordinates, axesLength,
                    angle, 0, 360, color, -1)

    if single_tubule:
        return canvas
    ###############
    # Second tubule
    ###############
    axesLength = (random.randrange(settings["axes_length_range"][0], settings["axes_length_range"][1]),
                  random.randrange(settings["axes_length_range"][0], settings["axes_length_range"][1]))  # w, h
    angle = random.randrange(settings["tubule_angle_range"][0], settings["tubule_angle_range"][1])
    thickness = random.randrange(settings["thickness_range"][0], settings["thickness_range"][1])
    center_coordinates = (center_coordinates[0] + 2 * axesLength[0],
                          center_coordinates[1] + random.randrange(settings["second_start_height_shift"][0],
                                                                   settings["second_start_height_shift"][1]))  # w, h
    # Draw tubule
    canvas = cv2.ellipse(canvas, center_coordinates, axesLength,
                         angle, 0, 360, color, thickness)
    # Fill tubule
    if fill == "yes" or (fill == "partial" and bool(random.getrandbits(1))):
        cv2.ellipse(canvas, center_coordinates, axesLength,
                    angle, 0, 360, color, -1)

    return canvas
