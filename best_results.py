import glob
import json
import re

root_dir = (
    r"/home/s0559816/Desktop/semantic-segmentation/logs/train_xxx/ocrnet.HRNet_Mscale/"
)

glolb_path = root_dir + "**/test*/logging.log"
logg_paths = glob.glob(glolb_path, recursive=True)


theDict = {}

for log in logg_paths:
    if "coll" in log:
        continue
    with open(log) as file:
        lines = file.readlines()
        for line in lines:
            path = log.replace(root_dir, "")
            if "num_classes" in line:
                hold = line.split(" ")
                try:
                    classes = int(hold[hold.index("num_classes") + 2])
                except:
                    continue
                if path in theDict.keys():
                    theDict[path]["num_classes"] = classes
                else:
                    theDict[path] = {"num_classes": classes}
            if "ego_trackbed" in line:
                test = re.search("[ \t]+\d+[ \t]+\w+([ \t]+\d+.\d+)+", line)
                if test:
                    hold = line.split(" ")
                    str_list = list(filter(None, hold))
                    iou = round(float(str_list[str_list.index("ego_trackbed") + 1]), 4)
                    if path in theDict.keys():
                        if "ego_trackbed" in theDict[path].keys():
                            if iou > theDict[path]["ego_trackbed"]:
                                theDict[path]["ego_trackbed"] = iou
                        else:
                            theDict[path]["ego_trackbed"] = iou
            if "ego_rails" in line:
                test = re.search("[ \t]+\d+[ \t]+\w+([ \t]+\d+.\d+)+", line)
                if test:
                    hold = line.split(" ")
                    str_list = list(filter(None, hold))
                    iou = round(float(str_list[str_list.index("ego_rails") + 1]), 4)
                    precision = round(
                        float(str_list[str_list.index("ego_rails") + 5]), 4
                    )
                    recall = round(float(str_list[str_list.index("ego_rails") + 6]), 4)
                    if path in theDict.keys():
                        if "ego_rails" in theDict[path].keys():
                            if iou > theDict[path]["ego_rails"]:
                                theDict[path]["ego_rails"] = iou
                                theDict[path]["precision"] = precision
                                theDict[path]["recall"] = recall
                                theDict[path]["F1"] = (precision + recall) / 2
                        else:
                            theDict[path]["ego_rails"] = iou
                            theDict[path]["precision"] = precision
                            theDict[path]["recall"] = recall
                            theDict[path]["F1"] = (precision + recall) / 2
            if "Mean" in line:
                test = re.search("\w+:[ \t]\d+.\d+", line)
                if test:
                    hold = line.split(" ")
                    str_list = list(filter(None, hold))
                    iou = round(float(str_list[str_list.index("Mean:") + 1]), 4)
                    if path in theDict.keys():
                        if "Mean" in theDict[path].keys():
                            if iou > theDict[path]["Mean"]:
                                theDict[path]["Mean"] = iou
                        else:
                            theDict[path]["Mean"] = iou
import os

for key in theDict.keys():
    if not "ego_rails" in theDict[key].keys():
        theDict[key]["ego_rails"] = 0

sortBy = "ego_rails"
# sortBy = "F1"
# sortBy = "ego_trackbed"
# sortBy = "Mean"
desc = False
sorted_dict = sorted(theDict.items(), key=lambda x: x[1][sortBy], reverse=desc)

print("Sorted by: {} ({})".format(sortBy, "descending" if desc else "ascending"))
print(json.dumps(sorted_dict, sort_keys=False, indent=4))
