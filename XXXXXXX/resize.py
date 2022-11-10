import concurrent.futures
import glob
import os
import random

from PIL import Image
from tqdm import tqdm

"""
This File creates a train and validation set with a ratio of 9-1
additionally a test dataset can be created from a seperate image path
all images are resized to a given size and changed to a .png file
"""


def resize_dataset(dataset, save_paths, resolution, interpolation):
    image = Image.open(dataset[0])
    mask = Image.open(dataset[1])

    if interpolation == "nearest":
        resample = Image.NEAREST
    elif interpolation == "box":
        resample = Image.BOX
    elif interpolation == "bilinear":
        resample = Image.BILINEAR
    elif interpolation == "hamming":
        resample = Image.HAMMING
    elif interpolation == "bicubic":
        resample = Image.BICUBIC
    elif interpolation == "lanczos":
        resample = Image.LANCZOS

    resized_image = image.resize((resolution[0], resolution[1]), resample=resample)

    resized_mask = mask.resize((resolution[0], resolution[1]), resample=Image.NEAREST)
    resized_image.save(save_paths[0])
    resized_mask.save(save_paths[1])


def gather_images(image_dir):
    """
    Gather all images in a directory with a .png extension

    Parameter: directory to images
    Returns: list of image paths
    """

    try:
        if os.path.exists(image_dir):
            search = os.path.join(image_dir, "*.png")
            images_glob = glob.glob(search)
            assert len(images_glob) != 0, f"Did not find any images in: {image_dir}"
    except BaseException:
        print(f"{image_dir} is not a valid path")

    return images_glob


def createDataset(list_subfolders, shuffle=False):
    images = []
    masks = []
    train_set = []
    test_set = []
    val_set = []

    for dir in list_subfolders:
        glob_list_img = []
        glob_list_mask = []
        for extention in ["png", "jpg", "jpeg"]:
            glob_path_img = os.path.join(dir, f"images/*.{extention}")
            glob_list_img.extend(glob.glob(glob_path_img))
            glob_path_mask = os.path.join(dir, f"masks/*.{extention}")
            glob_list_mask.extend(glob.glob(glob_path_mask))

        images = sorted(glob_list_img)
        masks = sorted(glob_list_mask)

        assert len(images) > 0, f"Did not find any images in {dir}"
        assert len(masks) > 0, f"Did not find any masks in {dir}"

        # make sure the amount of masks is equal to images
        if len(images) != len(masks):
            if len(images) > len(masks):
                for i in range(0, len(images)):
                    if os.path.basename(images[i]) not in masks[i]:
                        raise Exception(f"{images[i]} does not have a mask")
            elif len(images < len(masks)):
                for i in range(0, len(masks)):
                    if os.path.basename(masks[i]) not in os.path.basename(images[i]):
                        raise Exception(f"{masks[i]} does not have a mask")

        dataset = list(zip(images, masks))

        if shuffle:
            random.seed(1)
            random.shuffle(dataset)

            # check whether the correct mask was zipped to its image
            for i in range(len(dataset)):
                img_filename = os.path.splitext(os.path.basename(dataset[i][0]))[0]
                mask_filename = os.path.splitext(os.path.basename(dataset[i][1]))[0]
                assert (
                    img_filename == mask_filename
                ), f"Lists are not sorted the same {dataset[i][0]} != {dataset[i][1]}"

        if "train" in os.path.basename(dir):
            train_set.extend(dataset)
        if "test" in os.path.basename(dir):
            test_set.extend(dataset)
        if "val" in os.path.basename(dir):
            val_set.extend(dataset)

    return train_set, val_set, test_set


def assemble_dataset(root_dir, resolution, interpol):
    list_subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]

    # generate train-validation Dataset
    train, val, test = createDataset(list_subfolders, shuffle=False)

    save_dir = os.path.join(
        os.path.dirname(root_dir),
        f"{resolution[0]}-{resolution[1]}/",
        interpol,
    )

    for dir in ["train", "val", "test"]:
        # generate save paths
        img_save_dir = os.path.join(save_dir, dir, "images")
        mask_save_dir = os.path.join(save_dir, dir, "masks")
        for subdir in [img_save_dir, mask_save_dir]:
            if not os.path.exists(subdir):
                os.makedirs(subdir, exist_ok=True)

        if dir == "train":
            paths = train.copy()
        elif dir == "val":
            paths = val.copy()
        elif dir == "test":
            paths = test.copy()

        img_save = []
        mask_save = []
        for i in range(len(paths)):
            img_name = os.path.splitext(os.path.basename(paths[i][0]))[0]
            mask_name = os.path.splitext(os.path.basename(paths[i][1]))[0]
            img_save.append(
                os.path.join(
                    img_save_dir,
                    f"{img_name}.png",
                )
            )
            mask_save.append(
                os.path.join(
                    mask_save_dir,
                    f"{mask_name}.png",
                )
            )

        arguments2 = (
            paths,
            list(zip(img_save, mask_save)),
            [resolution] * len(paths),
            [interpol] * len(paths),
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(resize_dataset, *arguments2),
                    total=len(arguments2[0]),
                    desc=f"resizing and saving {dir} images",
                )
            )

    return save_dir


def main(dataset_dir, resolution, interpolation):
    if not os.path.exists(dataset_dir):
        raise Exception(f"{dataset_dir} is not a valid path")

    assert (
        resolution[0] > 0 and resolution[1] > 0
    ), f"{resolution} is not a valid resolution"
    assert (
        len(resolution) == 2
    ), f"{resolution} should consist of 2 values: width, height"

    for interpol in interpolation:
        assembled_path = assemble_dataset(dataset_dir, resolution, interpol)

        print(f"Dataset generated on: {assembled_path}")

    return assembled_path


if __name__ == "__main__":
    base_path = "/home/s0559816/Desktop/dataset2/base"
    resolution = (720, 304)
    interpolation = ["nearest", "bilinear"]

    main(base_path, resolution, interpolation)
