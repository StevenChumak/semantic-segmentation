import glob
import os


base_path = "/home/s0559816/Desktop/semantic-segmentation/logs/"

logs = glob.glob(os.path.join(base_path, "**/logging.log"), recursive=True)
global_dict = {}
global_best = 0
global_best_line = ''
global_best_log = None
for log in logs:
    with open(log, "r") as file:
        max_ego = 0
        bestLine = ""
        for line in file:
            if "ego_trackbed" in line:
                test = line.split()
                try:
                    hodl = float(test[2])
                except:
                    continue
                if max_ego < hodl:
                    max_ego = hodl
                    bestLine = line
    
    if global_best < max_ego:
        global_best = max_ego  
        global_best_line = bestLine
        global_best_log = log       
    
    global_dict[log] = {"iou":max_ego, "line":bestLine}
        
sorted = sorted(global_dict.items(), key=lambda item: item[1]["iou"], reverse=True)

print("Top 5")
for item in sorted[:5]:
    print(os.path.dirname(item[0]))
    print(item[1])