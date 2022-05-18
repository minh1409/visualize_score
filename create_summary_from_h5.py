from vsum_tools import *
from vasnet import *

import h5py
import numpy as np

h5_path = f"./traffic_dataset_2021-10-07 10:01:16.078811.h5"
h5_path = f"/current/traffic_dataset_2021-10-07 10 01 16.078811.h5"

model_file_path = '/working/model.pth.tar'

opened_h5 = h5py.File(h5_path)

all = []
summaries = {}
for video_id in ['4zSLT5pgQGU', 'c1EwbnIhHZg', 'E-rXC0tkWaM', 'GPONDzemTrM', 'mCwcunc7e1I', 'Z6DPz9aLHw4'] :
    f = opened_h5[video_id]

    # for i in f:
    #     print(f[i])

    
    changepoints, nfps, picks, features = list(map(np.array, [f['change_points'], f['n_frame_per_seg'], f['picks'], f['features'] ]))
    frameCount = f['n_frames'][()]
    print(video_id, frameCount, 'frames')
    import numpy as np
    aonet = AONet()

    aonet.initialize(f_len = len(features[0]))
    aonet.load_model(model_file_path)
    # print("load model successfull")
    predict = aonet.eval(features)
    # print(p)
    threshold = 0.5

    summary = generate_summary(predict, np.array(changepoints), int(frameCount),nfps, picks)
    summaries[video_id] = summary
    res = evaluate_summary(summary, np.array(f['user_summary']))

    all.append([video_id, frameCount, res[0], res[1], res[2]])

# from tabulate import tabulate
# print(tabulate(all, headers=['video id', 'total_frame','f1', 'precision', 'recall']))
print(*all, sep='\n')
print("Average f1 ", round(sum([i[2] for i in all])/len(all)*100,2), "%")
print("Average precision ", round(sum([i[3] for i in all])/len(all)*100,2), "%")