import os
import glob
from PIL import Image

path = "/home/s0559816/Desktop/dataset/assembled-1440-720-2/test/"
target_filetype ="jpeg"
destination_filetype = "png"

img_path = os.path.join(path, "images")

if os.path.exists(img_path):
    target = os.path.join(img_path, f"*.{target_filetype}")
    images = glob.glob(target)
    for image in images:
        image_name = os.path.splitext(os.path.basename(image))[0]
        destination = os.path.join(img_path, f"{image_name}.{destination_filetype}")
        # print(destination)
        target_image = Image.open(image)
        target_image.save(destination)
else:
    print("NAYY")
