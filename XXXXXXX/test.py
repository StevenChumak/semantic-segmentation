import pathlib

import torch


def load_checkpoint(checkpoint_path):
    checkpoint_path = pathlib.Path(checkpoint_path)
    if checkpoint_path.exists():
        checkpoint = torch.load(str(checkpoint_path), map_location=torch.device("cpu"))
    else:
        raise Exception(FileExistsError)

    return checkpoint


trained_weights = "/home/s0559816/Desktop/semantic-segmentation/logs/train_xxx/ocrnet.HRNet/64,64/16/_scarlet-skunk_2022.09.01_16.14/best_checkpoint_ep57.pth"
checkpoint = load_checkpoint(trained_weights)

checkpoint["label"] = {
    "background": {"id": 0, "trainId": 0},
    "left_trackbed": {"id": 48, "trainId": 0},
    "left_rails": {"id": 49, "trainId": 1},
    "ego_trackbed": {"id": 50, "trainId": 0},
    "ego_rails": {"id": 51, "trainId": 1},
    "right_trackbed": {"id": 52, "trainId": 0},
    "railhead": {"id": 53, "trainId": 1},
}
del checkpoint["ego_trackbed"]
del checkpoint["ego_rails"]
del checkpoint["mean_iu"]

# checkpoint["mean_iou"] = 85.52
# checkpoint["background_iou"] = 96.06
# checkpoint["railhead_iou"] = 74.97


torch.save(checkpoint, "railHeads.pth")

print(checkpoint)
