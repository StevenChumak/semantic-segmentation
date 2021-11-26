
import os
import yaml
import glob
import random
import concurrent.futures

from PIL import Image
from tqdm import tqdm
import cv2


def resize_dataset(dataset, save_paths, resolution):
    # image = Image.open(dataset[0])
    # mask = Image.open(dataset[1])

    # Resize images and masks
    # image = image.resize(resolution, resample=Image.NEAREST)
    # mask = mask.resize(resolution, resample=Image.NEAREST)

    # image.save(save_paths[0])
    # mask.save(save_paths[1])

    image = cv2.imread(dataset[0])
    mask = cv2.imread(dataset[1], cv2.IMREAD_GRAYSCALE)

    resized_image = cv2.resize(image, (resolution[0], resolution[1]), interpolation = cv2.INTER_NEAREST)
    resized_mask = cv2.resize(mask, (resolution[0], resolution[1]), interpolation = cv2.INTER_NEAREST)
    
    cv2.imwrite(save_paths[0], resized_image)
    cv2.imwrite(save_paths[1], resized_mask)


def assemble_dataset(root_dir, resolution, remove_path):
    list_subfolders = [
        f.path for f in os.scandir(root_dir) if f.is_dir()]

    rem = []
    for i, elem in enumerate(list_subfolders):
        for substring in remove_path:
            if substring in elem:
                rem.append(i)

    list_subfolders = [i for j, i in enumerate(list_subfolders) if j in rem]

    print(list_subfolders)

    images = []
    masks = []

    for dir in list_subfolders:
        img_dir = os.path.join(dir, "images_h")
        if not os.path.exists(img_dir):
            img_dir = os.path.join(dir, "images")

        mask_dir = os.path.join(dir, "new_gTruth_h")
        if not os.path.exists(mask_dir):
            mask_dir = os.path.join(dir, "new_gTruth")

        img_paths = glob.glob(os.path.join(img_dir, "*.png"))
        if not img_paths:
            img_paths = glob.glob(os.path.join(img_dir, "*.jpeg"))
        mask_paths = glob.glob(os.path.join(mask_dir, "*.png"))

        images.extend(img_paths)
        masks.extend(mask_paths)

    # make sure the amount of masks is equal to images
    assert len(images) == len(
        masks
    ), f"sum of images: {len(images)} is not equal to the sum of masks: {len(masks)}"

    dataset = list(zip(sorted(images), sorted(masks)))

    random.seed(1)
    random.shuffle(dataset)

    # make sure the mask corresponding to an image is the correct one
    for i in range(len(dataset)):
        img_filename = os.path.splitext(os.path.basename(dataset[i][0]))[0]
        mask_filename = os.path.splitext(os.path.basename(dataset[i][1]))[0]
        assert img_filename == mask_filename, f"Lists are not sorted the same {dataset[i][0]} != {dataset[i][1]}"

    save_dir = os.path.join(root_dir, f"assembled-{resolution[0]}-{resolution[1]}-2/test")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    data = {}
    data["image folder used:"] = list_subfolders

    with open(os.path.join(save_dir, "img_sources.yaml"), 'w+') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    save_img_dir = os.path.join(save_dir, "images")
    save_img_train_dir = os.path.join(save_img_dir, "trn")
    save_img_val_dir = os.path.join(save_img_dir, "val")

    save_mask_dir = os.path.join(save_dir, "masks")
    save_mask_train_dir = os.path.join(save_mask_dir, "trn")
    save_mask_val_dir = os.path.join(save_mask_dir, "val")

    # create any of the above directories, if they are missing
    for dir in [
        save_dir,
        save_img_dir,
        save_mask_dir,
        save_img_train_dir,
        save_img_val_dir,
        save_mask_train_dir,
        save_mask_val_dir,
    ]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    img_save = []
    mask_save = []

    counter = 0
    for i in range(len(dataset)):
        if counter < len(dataset) // 10:
            img_save.append(
                os.path.join(save_img_val_dir, os.path.basename(dataset[i][0]))
            )
            mask_save.append(
                os.path.join(save_mask_val_dir, os.path.basename(dataset[i][1]))
            )
            counter = counter + 1
        else:
            img_save.append(
                os.path.join(save_img_train_dir, os.path.basename(dataset[i][0]))
            )
            mask_save.append(
                os.path.join(save_mask_train_dir, os.path.basename(dataset[i][1]))
            )

    arguments = (dataset, list(zip(img_save, mask_save)), [resolution] * len(dataset))

    for i in range(len(arguments[0])):
        # check whether the filenames for image, image_mask, image's save path and image_mask's save path have all the same filename, ignoring image extentions
        assert (
            os.path.splitext(os.path.basename(arguments[0][i][0]))[0]
            == os.path.splitext(os.path.basename(arguments[0][i][1]))[0]
            == os.path.splitext(os.path.basename(arguments[1][i][0]))[0]
            == os.path.splitext(os.path.basename(arguments[1][i][1]))[0]
        ), f"Argument lists are not sorted the same {arguments[0][i][0]} != {arguments[0][i][1]} != {arguments[1][i][0]} != {arguments[1][i][1]}"

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(
            tqdm(executor.map(resize_dataset, *arguments), total=len(arguments[0]))
        )
    return results


def main():
    root_dir = "/home/s0559816/Desktop/dataset/"
    resolution = (1440, 720)
    remove_path = ["frame_to_map_matching_s21_unpersonal_measurement_2020-05-20_001", "new-package-2021.03.24-16.30.42-chunk-1_001"]

    assemble_dataset(root_dir, resolution, remove_path)


if __name__ == "__main__":
    main()
