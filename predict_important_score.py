from vsum_tools import *
from vasnet import *

model_file_path = '/working/model.pth.tar'


import json 
import sys
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image


input = json.load(sys.stdin)

# sampled_per_segment = input['segment_length_in_seconds'] * input['sampling_rate_in_fps']

# print(sampled_per_segment)
# quit()

batch_size = 50 #  number of frame to extract features at once

features = None

model_func, preprocess_func, target_size = model_picker('inceptionv3', 'max')



batch_frame = []

def do_one_batch_frame():
    global batch_frame
    global features
    batch_frame = np.vstack(batch_frame)
    batch_features = model_func.predict(batch_frame)

    if features is None:
        features = batch_features
    else:
        features = np.vstack((features, batch_features))
    
    batch_frame = []
for i in input['frame_list_as_png_file']:
    frame = cv2.imread(i)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, target_size).astype("float32")
    img_array = image.img_to_array(frame)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_func(expanded_img_array)

    batch_frame.append(preprocessed_img)

    if (len(batch_frame) == batch_size):
        do_one_batch_frame()
        print("extracted features for ", i )

if len(batch_frame) > 0:
    do_one_batch_frame()


aonet = AONet()

aonet.initialize(f_len = len(features[0]))
aonet.load_model(model_file_path)
# print("load model successfull")
predict = aonet.eval(features)

print(predict)
# quit()

# aggregate predicted score for segment by average score of all sampled frames in  that segment

sampled_per_segment = input['segment_length_in_seconds'] * input['sampling_rate_in_fps']
b = np.array_split(predict, input['number_of_sampled_frame'] // sampled_per_segment+1 )

print(predict.shape)
print([b.shape for b in b])

