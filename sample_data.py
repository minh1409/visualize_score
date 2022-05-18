
total_extracted_frame = int("002220")

import glob
frames_png_list = sorted( glob.glob('test_save_load_frame/frame_*.png') )

import json

print(json.dumps({
    'number_of_sampled_frame' : total_extracted_frame,
    'sampling_rate_in_fps' : 2,
    'segment_length_in_seconds' : 2,
    'frame_list_as_png_file' : frames_png_list
}, indent=2))