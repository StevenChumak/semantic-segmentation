import os
import glob
from PIL import Image
import shutil
import numpy as np


train = "/home/s0559816/Desktop/dataset/assembled-1440-720/"


images = os.path.join(train, 'images')
images_list_subfolders = [
        f.path for f in os.scandir(images) if f.is_dir()]

masks = os.path.join(train, 'masks')
masks_list_subfolders = [
        f.path for f in os.scandir(masks) if f.is_dir()]

img_paths = []
mask_paths = []

for i in range(len(images_list_subfolders)):
    img_paths.extend(glob.glob(os.path.join(images_list_subfolders[i], "*.png")))
    mask_paths.extend(glob.glob(os.path.join(masks_list_subfolders[i], "*.png")))

for i in range(len(img_paths)):

    # open image and convert mask to binary, turning every non-zero value to 1
    mask = Image.open(mask_paths[i])
    data = np.array(mask).astype(np.uint8)
    # data[data==48] = 1
    # data[data==49] = 2
    # data[data==50] = 3
    # data[data==51] = 4
    # data[data==52] = 5
    # data[data==53] = 6
    mask = Image.fromarray(data)
    
    # generate save paths and create dirs if missing
    test = os.path.basename(os.path.dirname(os.path.dirname(img_paths[i])))
    test2 = os.path.basename(os.path.dirname(os.path.dirname(mask_paths[i])))
    test3 = os.path.basename(os.path.dirname(img_paths[i]))
    test4 = os.path.basename(os.path.dirname(mask_paths[i]))

    img_save_path = os.path.join(train, "reducedToSingleDigit/", test, test3)
    mask_save_path = os.path.join(train, "reducedToSingleDigit/", test2, test4)

    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)
    
    # copy image files and save masks in new path
    shutil.copyfile(img_paths[i], os.path.join(img_save_path, os.path.basename(img_paths[i])))
    mask.save(os.path.join(mask_save_path, os.path.basename(mask_paths[i])))
