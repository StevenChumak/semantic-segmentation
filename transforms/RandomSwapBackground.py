from typing import Union, List, Tuple
import pathlib
import random

import cv2
import numpy as np
from PIL import Image
from numpy.lib.type_check import imag


def swap_background(
    image,
    mask,
    background,
    normalized = False,
    type = "float32",
):
    """
    Blend in random background image.
    Separation of foreground and background is defined
    by the mask.
    Colored image information is expected and provided in
    RGBC order.

    :param fg_arr: Foreground image as numpy array.
    :param msk_arr: Mask image as numpy array.
    :param bg_dir_pth: Path to background image directory.
    :param bg_ext: Background image file extension.
    :param p: Probability to apply augmentation.
    :return: Augmented image as numpy array.
    """

    assert type == "float32" or type == "uint8", f"Unsupported type {type}"
    # Random number to decide if augmentation is applied.
    rand_nr: float = random.random()

    # Apply augmentation.
    if rand_nr < 0.9:

        fg_arr = np.array(image)
        # Convert RGB input to cv2 native color order
        fg_arr = cv2.cvtColor(fg_arr, cv2.COLOR_RGB2BGR)

        # transform mask into numpy array and combine all lables together
        msk_arr = np.array(mask).astype(bool).astype(np.uint8)

        bg_arr = np.array(background)
        fg_size = tuple(reversed(fg_arr.shape[:2]))
        bg_arr = cv2.resize(bg_arr, fg_size, interpolation=cv2.INTER_CUBIC)
        # Invert mask and spread contrast
        msk_arr = 255 - msk_arr * 255

        # Calculate distance to label and min max scale
        # distance.
        dist_arr: np.ndarray
        dist_arr, label = cv2.distanceTransformWithLabels(msk_arr, cv2.DIST_L2, 5)
        # TODO: Clean solution for zero division case.
        if (dist_arr.max() - dist_arr.min()) > 0:
            dist_arr = (dist_arr - dist_arr.min()) / (dist_arr.max() - dist_arr.min())
        else:
            file_path = pathlib.Path(__file__).resolve()
            print(f"{file_path}: Dist_arr <= 0!")
            return image
        dist_arr = np.stack([dist_arr] * 3, axis=2)

        # Normalize foreground and background.
        bg_arr = bg_arr / 255.0

        if not normalized:
            fg_arr = fg_arr / 255.0

        # Blend foreground and background.
        dist_arr = 1 - (1 - dist_arr) ** 3
        blended: np.ndarray = (1 - dist_arr) * fg_arr + dist_arr * bg_arr

        if type == "uint8":
            # Back to uint8 range.
            blended = (blended * 255).astype(np.uint8)
        if type == "float32":
            # Back to float32 range.
            blended = (blended).astype(np.float32)


        # Convert cv2 native color order to RGB
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        blended = Image.fromarray(blended).convert("RGB")

    # Do not apply augmentation
    else:
        blended = image

    return blended
