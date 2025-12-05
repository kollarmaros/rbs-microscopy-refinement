import random
import math

import numpy as np
import cv2
from scipy import ndimage

from tubules import create_outer_pair, create_central_pair
from texture import texture_by_objects
from utils import (create_mask_by_hull, merge_segmentation_mask,
                   coordinates_smoothing, get_edges_positions, crop_to_object)


def create_axoneme(settings):
    canvas_dimension = (settings["canvas_shape"], settings["canvas_shape"])
    mask_dimension = (settings["canvas_shape"], settings["canvas_shape"], 3)
    canvas = np.uint8(np.ones(canvas_dimension) * 255)
    canvas_outer_pairs = np.uint8(np.ones(canvas_dimension) * 255)
    center_coordinates = (settings["canvas_shape"]//2, settings["canvas_shape"]//2)
    segmentation_canvas = np.zeros(mask_dimension)

    outer_pairs_number = settings["outer_pairs_number"]
    outer_pair_radius = random.randrange(settings["outer_pair_radius_range"][0],
                                         settings["outer_pair_radius_range"][1])
    outer_pair_step = 2 * math.pi / outer_pairs_number
    draw_central_pair = True
    first_outer_as_central = False

    # Setup CCD defects
    if settings["type"] == "CCD":
        if settings["variation"] == "no_central_pair":
            draw_central_pair = False
        elif settings["variation"] == "single_tubule":
            settings["central_pair"]["defect"] = "single_tubule"
        elif settings["variation"] == "moved_to_outer":
            draw_central_pair = False
            outer_pairs_number += 1
            outer_pair_step = 2 * math.pi / outer_pairs_number  # recalculate step for add new pair
            if random.choice([True, False]):  # possibility to create single tubule when moved to outer tubules
                settings["central_pair"]["defect"] = "single_tubule"
            first_outer_as_central = True
        elif settings["variation"] == "multiple":
            settings["central_pair"]["defect"] = "multiple"
        else:
            raise Exception(f"Unknown 'CCD' variation: {settings['variation']}")

    if settings["type"] == "transposition":
        outer_pairs_number -= 1
        if settings["variation"] == "with_space":
            pass
        elif settings["variation"] == "without_space":
            outer_pair_step = 2 * math.pi / outer_pairs_number
            outer_pair_radius = random.randrange(settings['transposition']["transposition_outer_pair_radius_range"][0],
                                                 settings['transposition']["transposition_outer_pair_radius_range"][1])
        else:
            raise Exception(f"Unknown 'transposition' variation: {settings['variation']}")

    # Central pair:
    if draw_central_pair:
        central_pair = create_central_pair(settings['central_pair'], settings["fill_tubules"])
        height, width = central_pair.shape
        start_height = center_coordinates[0] - height // 2
        start_width = center_coordinates[1] - width // 2
        canvas[start_height:start_height+height, start_width:start_width+width] = central_pair

        # create segmentation mask for central pair
        canvas_revert = 255 - canvas
        _, canvas_revert_thresh = cv2.threshold(canvas_revert, 64, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(canvas_revert_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(segmentation_canvas, contours, -1, (0, 255, 0), -1)

    disarranges_indexes = []
    if settings["type"] == "disarranged" and settings["variation"] == "partial":
        number_of_disarranges = random.randint(1, outer_pairs_number)

        for x in range(number_of_disarranges):
            while len(disarranges_indexes) == x:
                index = random.randint(0, outer_pairs_number - 1)
                if index not in disarranges_indexes:
                    disarranges_indexes.append(index)

    single_indexes = []
    if settings["type"] == "single_tubule":
        number_of_single_tubules = random.randint(1, settings["single_tubule"]["max_tubules"])
        for x in range(number_of_single_tubules):
            while len(single_indexes) == x:
                index = random.randint(0, outer_pairs_number - 1)
                if index not in single_indexes:
                    single_indexes.append(index)

    # Outer pairs
    for pair_number in range(outer_pairs_number):
        t = pair_number * outer_pair_step
        x = round(outer_pair_radius * math.cos(t) + center_coordinates[0])  # width
        y = round(outer_pair_radius * math.sin(t) + center_coordinates[1])  # height

        if pair_number == 0 and first_outer_as_central:
            out_pair = create_central_pair(settings['central_pair'], settings["fill_tubules"])
        else:
            single_tubule = False
            if pair_number in single_indexes:
                single_tubule = True
            out_pair = create_outer_pair(settings['outer_pair'], settings["fill_tubules"], single_tubule=single_tubule)
            out_pair = np.uint8(np.clip(ndimage.rotate(out_pair, 90 - t * 180 / math.pi, cval=255), 0, 255))

        if (settings["type"] == "disarranged" and settings["variation"] == "partial") and \
           (pair_number in disarranges_indexes):
            # change radius for tubule
            movement_ratio = random.randrange(settings["disarranged"]["movement_ratio_range"][0],
                                              settings["disarranged"]["movement_ratio_range"][1]) / 100
            new_radius = outer_pair_radius + random.choice([1, -1]) * int(outer_pair_radius * movement_ratio)
            x = round(new_radius * math.cos(t) + center_coordinates[0])  # width
            y = round(new_radius * math.sin(t) + center_coordinates[1])  # height

            # change rotation of tubule
            rotation_angle = random.randrange(settings["disarranged"]["angle_difference_range"][0],
                                              settings["disarranged"]["angle_difference_range"][1])
            out_pair = np.uint8(np.clip(ndimage.rotate(out_pair, rotation_angle, cval=255), 0, 255))

        height, width = out_pair.shape

        start_height = y - height // 2
        start_width = x - width // 2

        tmp_canvas = np.uint8(np.ones(canvas_dimension) * 255)
        while True:
            try:
                tmp_canvas[start_height:start_height+height, start_width:start_width+width] = out_pair
                break
            except ValueError:
                if settings["type"] != "disarranged":
                    raise ValueError("not in disarranged type")
                movement_ratio = random.randrange(settings["disarranged"]["movement_ratio_range"][0],
                                                  settings["disarranged"]["movement_ratio_range"][1]) / 100
                new_radius = outer_pair_radius + random.choice([1, -1]) * int(outer_pair_radius * movement_ratio)
                x = round(new_radius * math.cos(t) + center_coordinates[0])  # width
                y = round(new_radius * math.sin(t) + center_coordinates[1])  # height
                start_height = y - height // 2
                start_width = x - width // 2

        canvas_outer_pairs = cv2.bitwise_and(canvas_outer_pairs, tmp_canvas)

    # create segmentation mask for outer pair
    canvas_revert = 255 - canvas_outer_pairs
    _, canvas_revert_thresh = cv2.threshold(canvas_revert, 64, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(canvas_revert_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(segmentation_canvas, contours, -1, (0, 255, 0), -1)

    # Merge central pair and outer pairs canvases into one canvas
    canvas = cv2.bitwise_and(canvas, canvas_outer_pairs)

    # Texture the axoneme of cilia
    if settings["add_texture"]:
        noise_setup = settings["central_noise"]
        canvas = texture_by_objects(noise_setup, canvas.astype(np.uint8))

    canvas = canvas.astype(np.uint8)
    axoneme_mask = create_mask_by_hull(canvas).astype(np.uint8)
    segmentation_canvas[:, :, 2] = axoneme_mask * 255

    return canvas.astype(np.uint8), segmentation_canvas.astype(np.uint8)


def add_membrane(cilium, segmentation_mask, settings,
                 number_of_nodes=23, additional_canvas_size=15):
    """
    Add membrane to the cilia.

    Args:
        cilia (np.array): sketch of the axoneme
        segmentation_mask (np.array): segmentation mask
        settings (dict): settings
        number_of_nodes (int): number of nodes for membrane
        additional_canvas_size (int): Size added to the canvas to prevent cutting the lines at the edges
    """
    axoneme_mask = (segmentation_mask[:, :, 2] / 255).astype(np.uint8)
    cnts, _ = cv2.findContours(axoneme_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    if len(cnts) == 1:
        (x_center, y_center), radius = cv2.minEnclosingCircle(cnts[0])
        radius = int(radius + 0.15 * radius)
        radius_random_max_part = int(radius * 0.2)
    else:
        print(f"{len(cnts)} contours found rather then only one")
        cilia_diameter = max(cilium.shape[:2])
        x_center = cilium.shape[0] // 2
        y_center = cilium.shape[1] // 2
        radius = int(cilia_diameter * 0.6)  # diameter // 2 + 0.2 = 0.7 diameter
        radius_random_max_part = int(cilia_diameter * 0.1)
    # TODO: improve randomness - add can be more, however weighting based on the distance to zero and
    # smaller probability for negative values ....
    random_step_part_ratio = 2 / 5
    center_coordinates = [0, 0]

    x_coordinates = []
    y_coordinates = []
    step = 2 * math.pi / number_of_nodes
    step_max_random_part = step * random_step_part_ratio
    first = True
    for node_number in range(number_of_nodes):
        step_random_part = random.choice([-1, 1]) * random.random() * step_max_random_part
        t = node_number * step + step_random_part

        radius_random_part = random.random() * radius_random_max_part

        x = round((radius + radius_random_part) * math.cos(t) + center_coordinates[0]) + x_center  # width
        y = round((radius + radius_random_part) * math.sin(t) + center_coordinates[1]) + y_center  # height

        if first:
            first = False
            first_point = [x, y]

        x_coordinates.append(x)
        y_coordinates.append(y)

    x_coordinates.append(first_point[0])
    y_coordinates.append(first_point[1])

    # Smoothing
    xi, yi, x_movement, y_movement, new_x_shape, new_y_shape = coordinates_smoothing(x_coordinates, y_coordinates,
                                                                                     additional_canvas_size)

    # new shape because of lines outside the original canvas
    cilium_w_membrane = np.ones((new_y_shape + 2 * additional_canvas_size,
                                 new_x_shape + 2 * additional_canvas_size)) * 255
    cilium_w_membrane[y_movement: y_movement + cilium.shape[0], x_movement: x_movement + cilium.shape[1]] = cilium
    segmentation_mask_w_axiom = np.zeros((new_y_shape + 2 * additional_canvas_size,
                                          new_x_shape + 2 * additional_canvas_size,
                                          3))
    segmentation_mask_w_axiom[y_movement: y_movement + cilium.shape[0],
                              x_movement: x_movement + cilium.shape[1]] = segmentation_mask

    color = random.randrange(settings["membrane_color_range"][0],
                             settings["membrane_color_range"][1] + 1)
    thickness = random.randrange(settings["membrane_thickness_range"][0],
                                 settings["membrane_thickness_range"][1] + 1)
    pts = np.transpose(np.array([xi, yi]), (1, 0)).reshape((-1, 1, 2)).astype(int)
    cilium_w_membrane = cv2.polylines(cilium_w_membrane, [pts], True, color, thickness)
    segmentation_membrane = np.zeros_like(cilium_w_membrane)
    segmentation_membrane = cv2.polylines(segmentation_membrane, [pts], True, 255, thickness)
    segmentation_membrane = cv2.fillPoly(segmentation_membrane, [pts], 255)
    segmentation_mask_w_axiom[:, :, 0] = segmentation_membrane

    cilium_w_membrane = cv2.resize(cilium_w_membrane, (settings["canvas_shape"],
                                                       settings["canvas_shape"])).astype(np.uint8)
    segmentation_mask_w_axiom = cv2.resize(segmentation_mask_w_axiom,
                                           (settings["canvas_shape"], settings["canvas_shape"])).astype(np.uint8)

    axoneme_mask = (segmentation_mask_w_axiom[:, :, 2] / 255).astype(np.uint8)
    if settings["add_texture"]:
        noise_setup = settings["inside_noise"]
        cilium_w_membrane = texture_by_objects(noise_setup, cilium_w_membrane, negative_mask=1-axoneme_mask)

    return cilium_w_membrane, segmentation_mask_w_axiom


def create_whole_cilia(settings, resize=None):
    edge_size = settings["canvas_shape"]  # size should be the same

    # Create axoneme
    axoneme, segmentation_mask = create_axoneme(settings)
    axoneme, segmentation_mask = crop_to_object(axoneme, segmentation_mask)

    # Add membrane
    cilium, segmentation_mask = add_membrane(axoneme, segmentation_mask, settings)

    # Unequal resize for deformation
    resize_x = random.randrange(settings["final_randomization"]["unequal_resize_range"][0],
                                settings["final_randomization"]["unequal_resize_range"][1]) / 100
    cilium = cv2.resize(cilium, (0, 0), fx=resize_x, fy=1)
    segmentation_mask = cv2.resize(segmentation_mask, (0, 0), fx=resize_x, fy=1)

    # rotation
    angle = random.randrange(settings["final_randomization"]["rotation_range"][0],
                             settings["final_randomization"]["rotation_range"][1])
    cilium = np.uint8(np.clip(ndimage.rotate(cilium, angle, cval=255), 0, 255))
    segmentation_mask = np.uint8(np.clip(ndimage.rotate(segmentation_mask, angle, cval=0), 0, 255))

    # Equal resize to make cilium smaller
    if resize is None:
        resize = random.randrange(settings["final_randomization"]["equal_resize_range"][0],
                                  settings["final_randomization"]["equal_resize_range"][1]) / 100
    cilium = cv2.resize(cilium, (0, 0), fx=resize, fy=resize)
    segmentation_mask = cv2.resize(segmentation_mask, (0, 0), fx=resize, fy=resize)

    height, width = cilium.shape
    # Rescale so that longest edge is fixed to predefined edge_size
    if height > edge_size or width > edge_size:
        bigger = max(height, width)
        scale = edge_size / bigger

        cilium = cv2.resize(cilium, (0, 0), fx=scale, fy=scale)
        segmentation_mask = cv2.resize(segmentation_mask, (0, 0), fx=scale, fy=scale)

    height, width = cilium.shape

    height_diff = edge_size - height
    width_diff = edge_size - width
    height_start = 0
    width_start = 0
    if height_diff != 0:
        height_start = random.randrange(0, height_diff)
    if width_diff != 0:
        width_start = random.randrange(0, width_diff)

    final_cilium = np.ones((edge_size, edge_size)) * 255
    final_segmentation_mask = np.zeros((edge_size, edge_size, 3))

    final_cilium[height_start:height_start+height, width_start:width_start+width] = cilium
    final_segmentation_mask[height_start:height_start+height, width_start:width_start+width] = segmentation_mask

    final_segmentation_mask = np.where(final_segmentation_mask > 127, 255, 0).astype(np.uint8)

    return final_cilium, final_segmentation_mask


def randomize_cilium(settings):
    settings["central_noise"]["shape"] = settings["canvas_shape"]
    settings["inside_noise"]["shape"] = settings["canvas_shape"]
    settings["outside_noise"]["shape"] = settings["canvas_shape"]

    resize = random.randrange(settings["final_randomization"]["equal_resize_range"][0],
                              settings["final_randomization"]["equal_resize_range"][1]) / 100
    final_cilium, final_segmentation_mask = create_whole_cilia(settings, resize)

    edge_size = settings["canvas_shape"]  # size should be the same

    # Crop so that part of the axiom would be missing
    if settings["final_randomization"]["close_fit_probability"] > random.randint(0, 99):
        top, bottom, left, right = get_edges_positions(final_cilium)
        top_addition_crop_ratio = random.randrange(settings["final_randomization"]["close_fit_crop_ratio"][0],
                                                   settings["final_randomization"]["close_fit_crop_ratio"][1]) / 100
        bottom_addition_crop_ratio = random.randrange(settings["final_randomization"]["close_fit_crop_ratio"][0],
                                                      settings["final_randomization"]["close_fit_crop_ratio"][1]) / 100
        close_height = bottom-top

        left_addition_crop_ratio = random.randrange(settings["final_randomization"]["close_fit_crop_ratio"][0],
                                                    settings["final_randomization"]["close_fit_crop_ratio"][1]) / 100
        right_addition_crop_ratio = random.randrange(settings["final_randomization"]["close_fit_crop_ratio"][0],
                                                     settings["final_randomization"]["close_fit_crop_ratio"][1]) / 100
        close_width = right-left

        final_cilium = final_cilium[top + int(close_height * top_addition_crop_ratio):
                                  bottom - int(close_height * bottom_addition_crop_ratio),
                                  left + int(close_width * left_addition_crop_ratio):
                                  right - int(close_height * right_addition_crop_ratio)]
        final_cilium = cv2.resize(final_cilium, (edge_size, edge_size))

        final_segmentation_mask = final_segmentation_mask[top + int(close_height * top_addition_crop_ratio):
                                                          bottom - int(close_height * bottom_addition_crop_ratio),
                                                          left + int(close_width * left_addition_crop_ratio):
                                                          right - int(close_height * right_addition_crop_ratio)]

        final_segmentation_mask = cv2.resize(final_segmentation_mask, (edge_size, edge_size))
    # Add additional cilia to the image
    elif (settings["final_randomization"]["multiple_cilia"] and
          settings["final_randomization"]["multiple_cilia_probability"] > random.randint(0, 99)):
        no_extra_cilia = random.randrange(1, settings["final_randomization"]['no_additional_cilia'] + 1)
        for x in range(no_extra_cilia):
            additional_cilium, additional_mask = create_whole_cilia(settings, resize=resize)

            radius = settings["final_randomization"]["multiple_cilia_radius"]
            radius_increment = settings["final_randomization"]["multiple_cilia_radius_increment"]
            angle = random.randrange(0, 359)

            merge_to_one_image = False
            for i in range(10):  # TODO: add to configuration - max attempts to move
                x = int(radius * math.cos(math.radians(angle)))
                y = int(radius * math.sin(math.radians(angle)))

                radius += radius_increment

                additional_cilium_moved = np.ones_like(additional_cilium) * 255
                additional_mask_moved = np.zeros_like(additional_mask)

                height, width = final_cilium.shape

                if x >= 0:
                    new_w_start = x
                    new_w_end = width
                    old_w_start = 0
                    old_w_end = width - x
                else:
                    new_w_start = 0
                    new_w_end = width + x
                    old_w_start = -x
                    old_w_end = width

                if y >= 0:
                    new_h_start = y
                    new_h_end = height
                    old_h_start = 0
                    old_h_end = height - y
                else:
                    new_h_start = 0
                    new_h_end = width + y
                    old_h_start = -y
                    old_h_end = width

                shape1 = additional_cilium_moved[new_h_start:new_h_end, new_w_start:new_w_end].shape
                shape2 = additional_cilium[old_h_start:old_h_end, old_w_start:old_w_end].shape
                if shape1 != shape2:
                    continue

                additional_mask_moved[new_h_start:new_h_end,
                                      new_w_start:new_w_end] = additional_mask[old_h_start:old_h_end,
                                                                               old_w_start:old_w_end]

                # Check if the new cilia does not overlap with the original cilia
                merged_mask_original = merge_segmentation_mask(final_segmentation_mask)
                merged_mask_additional = merge_segmentation_mask(additional_mask_moved)
                if np.sum(merged_mask_original * merged_mask_additional) == 0:
                    merge_to_one_image = True
                    break

            if merge_to_one_image:
                additional_cilium_moved[new_h_start:new_h_end,
                                       new_w_start:new_w_end] = additional_cilium[old_h_start:old_h_end,
                                                                                 old_w_start:old_w_end]

                final_cilium = final_cilium * (1 - merged_mask_additional) + \
                    merged_mask_additional * additional_cilium_moved
                final_segmentation_mask = final_segmentation_mask + additional_mask_moved

    if settings["add_texture"]:
        noise_setup = settings["outside_noise"]
        final_cilium = texture_by_objects(noise_setup, final_cilium.astype(np.uint8), final_segmentation_mask)

    for x in range(final_segmentation_mask.shape[2]):
        single_channel = final_segmentation_mask[:, :, x]
        single_channel[single_channel >= 127] = 255
        single_channel[single_channel < 127] = 0
        final_segmentation_mask[:, :, x] = single_channel

    return final_cilium.astype(np.uint8), final_segmentation_mask.astype(np.uint8)
