
from operator import getitem

image_stats = {
    'nlb_winter_000243_stacked.png': 
        {
            'iou':  [0.9904383420944214, 0.09581395238637924, 0.1029900312423706, 0.0],
            'a': 1,
            'b': 2,
            'c': 3,
            'd': 4,
        },
        
    'nlb_summer_001301_stacked.png': 
        {
            'iou': [0.9705694317817688, 0.22962476313114166, 0.01498127356171608, 0.0], 
            'a': 1,
            'b': 2,
            'c': 3,
            'd': 4,
        }
    }
# for key, value in image_stats.items():
    # print(value[1])
    # print()
    
stats = dict(sorted(image_stats.items(), key=lambda x:getitem(x[1], "iou")[1]))
print(stats.keys())