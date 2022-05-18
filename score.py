"""IMPORT LIBRARIES"""
import numpy as np
import tensorflow.keras.backend
from vsum_tools import *
from vasnet import *
import shutil
import os
import re
import h5py
import cv2
import sys
import skvideo.io
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import KDTree
import csv
from sklearn.cluster import KMeans
from pytube import YouTube
from tkinter import *
import threading
import time

"""------------------------------------------------------------------------------------------------------------------"""

"""DEFINE CONSTANT"""

PARENTDIR = 0
FILENAME = 1
FILEDIR = 2

"""------------------------------------------------------------------------------------------------------------------"""

"""READING CONSTANTS"""

# File config dùng để lưu các đường dẫn, hằng số và kí hiệu phân cách giữa các thư mục và tệp tin
# Đường dẫn: - input_directory là đường dẫn đến thư mục chứa các video cần được tóm tắt
#            - output_directory là đường dẫn đến thư mục chứa các video đã được tóm tắt
#            - gt_directory là đường dẫn đến thư mục chứa GT của tập test input
#            - saved_model_directory là đường dẫn đến file weight của model
#            -
# Các hằng số: - segment_length là độ dài mỗi đoạn cắt
#              - sampling_rate là lượng frame model lấy mỗi giây
#              - model_name là loại mô hình được sử dụng để chiết xuất đặc trưng ảnh:
#                   + inceptionv3
#                   + inceptionresnetv2
#                   + mobilenet
#                   + vgg19
#                   + vgg16
#                   + resnet50
#                   + xception
#                   + nasnetlarge
#                   + efficientnetb3
#                   + efficientnetb5
#                   + efficientnetb7
#              - pooling_type là loại lớp lấy mẫu được sử dụng
#              - dirsep là ký hiệu phân cách giữa các thư mục và tệp tin:
#                   + Sử dụng kí hiệu \ cho Windows
#                   + Sử dụng kí hiệu / cho Linux

config_file = open("config.txt", "r")
temp = dict()

# Giải thuật đọc file config sử dụng một mảng băm (dict) để cho quá trình nhập config bỏ qua thứ tự nhập các hằng số,...
while True:
    line = config_file.readline()
    if not line:
        break
    line = line.replace(' ', '$').replace('=', '$').replace('\n', '').split(sep='$')
    temp.update({line[0] : line[1]})

#Cập nhật các hằng số, đường dẫn và kí hiệu phân cách
segment_length = int(temp['segment_length'])
sampling_rate = int(temp['sampling_rate'])
input_directory = temp['input_directory']
output_directory = temp['output_directory']
gt_directory = temp['gt_directory']
model_file_path = temp['saved_model_directory']
model_name = temp['model_name']
pooling_type = temp['pooling_type']
dirsep = temp['dirsep']

#root_directory là đường dẫn đến thư mục chứa mã nguồn
root_directory = os.getcwd() + dirsep

"""------------------------------------------------------------------------------------------------------------------"""

"""INITIALIZING"""

#Khởi tạo mô hình chiết xuất đặc trưng
model_func, preprocess_func, target_size = model_picker(model_name, pooling_type)

"""------------------------------------------------------------------------------------------------------------------"""

"""GET ALL FILE PATH WITH GIVEN EXTENSION"""

def get_all_file_path(directory, file_extension):
    all_filepath = [] # Đây là mảng lưu lại tất cả các đường dẫn chứa file có đuôi file_extension tại thư mục directory
    for root, dirs, files in os.walk(directory): # Vòng lặp để quét tất cả các file trong thư mục
        for file in files:
            for ext in file_extension:
                if file.endswith(ext): # Xuất hiện bug có kí hiệu phân cách và không có, dòng if để chuẩn hóa đường dẫn
                    if root[len(root) -  1] != dirsep:
                        all_filepath.append((root + dirsep, file, root + dirsep + file))
                    else:
                        all_filepath.append((root, file, root + file))
    return all_filepath

"""------------------------------------------------------------------------------------------------------------------"""

"""SUMMARIZE VIDEO FUNCTION"""
def remove_extension(filename): # Là hàm để loại bỏ đuôi trong tên file, chủ yếu dùng để tạo đường dẫn đến thư muc groundTrue
    temp = list(filename)
    while temp[len(temp) - 1] != '.':
        temp.pop()
        if len(temp) == 0:
            return filename
    temp.pop()
    return "".join(temp)

def take_file_name(x): # Chỉ là một hàm trung gian để phục vụ cho việc so sánh trong hàm sort
    return (len(x[FILENAME]), x[FILENAME]) # bởi vì GT có nhiều user nên cần được sắp cho đúng theo trình tự thời gian

def frame_list_to_features(frame_list, model_func, preprocess_func, target_size):
    for i in range(0, len(frame_list)):
        frame_list[i] = cv2.cvtColor(frame_list[i], cv2.COLOR_BGR2RGB)
        frame_list[i] = cv2.resize(frame_list[i], target_size).astype("float32")
        img_array = image.img_to_array(frame_list[i])
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_func(expanded_img_array)
    frame_list = np.asarray(frame_list)
    features = model_func.predict(frame_list, batch_size=32)
    print(features)
    print(features.mean())
    print(features.max())
    print(features.min())
    return features

def write_frame(frame, str, isGT=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Use putText() method for
    # inserting text on video
    color = None
    if isGT == False:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    cv2.putText(frame,
                str,
                (50, 50),
                font, 1,
                color,
                2,
                cv2.LINE_4)
    return None

def draw(mat, position, radius=0):
    for i in range(radius * -1, radius + 1):
        for j in range(radius * -1, radius + 1):
            try:
                if i**2 + j**2 > radius:
                    continue
                mat[position[0] + i][position[1] + j][0] = 0
                mat[position[0] + i][position[1] + j][1] = 0
                mat[position[0] + i][position[1] + j][2] = 255
            except:
                pass
    return

def draw_graph(predict, frame):
    graph = np.zeros((100, frame[0].shape[1], 3), dtype=np.uint8)
    # print(frame[0].shape, graph.shape)
    # print(frame[0])
    for i in range(min(len(frame), len(predict))):
        height = 100 - int(round(predict[i] * 100))
        prog = int(round(i / len(frame) * frame[0].shape[1]))
        for j in range(0, height):
            draw(graph, (j, prog), radius=1)
        # cv2.imshow("graph", graph)
        # cv2.imshow("frame", frame[i])
        # cv2.waitKey(200)
        frame[i] = np.concatenate((frame[i], graph), axis=0)
    # print(frame[0].shape)
    # print(frame[0])
    return None

def summarize_video(video_path, output_path, filename):
    # Đây là code của thầy An
    video = cv2.VideoCapture(video_path)
    got, frame = video.read()
    print(got, len(frame))

    fps = (video.get(cv2.CAP_PROP_FPS))
    frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if len(frame)/fps < 120:
        sampling_rate = 15
    size_param = "%dx%d" % size
    padding = lambda i: '0' * (6 - len(str(i))) + str(i)
    print(fps, frameCount, size)

    changepoints = [(i, i + int(segment_length * fps) - 1) for i in
                    range(0, int(frameCount), int(segment_length * fps))]
    nfps = [x[1] - x[0] + 1 for x in changepoints]
    picks = []
    i = 0
    while True:
        picks += [int(fps * i + x * int(fps // sampling_rate)) for x in range(sampling_rate)]
        i += 1
        if picks[-1] > frameCount:
            break
    picks = [i for i in picks if i < frameCount]
    print(picks)
    frames, features = extract_frame_and_features4video(model_func, preprocess_func, target_size, picks, video_path)

    # print("FEATURES: ")
    # print(features)
    # print(features.shape)
    # print(evaluate_error(features, features))
    # print("END:")

    aonet = AONet()
    aonet.initialize(f_len=len(features[0]))
    aonet.load_model(model_file_path)
    print("load model successfull")
    predict = aonet.eval(features)
    print("PREDICT:", predict)
    i = 0
    for i in range(min(len(frames), len(predict))):
        x = str(round(predict[i], 2))
        write_frame(frames[i], x)
        # cv2.imshow('test', frames[i])
        # cv2.waitKey(30)
    draw_graph(predict, frames)

    i = 0
    New_Window()
    while True:
        if check_quit == True:
            break
        if i < 0:
            if i + incre == 0:
                i += incre
            else:
                continue
        if i == min(len(frames), len(predict)):
            if i + incre < min(len(frames), len(predict)):
                i += incre
            else:
                continue
        while Pause_check == True:
            pass
        cv2.imshow('Video', frames[i])
        cv2.waitKey(speed)
        i += incre
    return predict

"""------------------------------------------------------------------------------------------------------------------"""
speed = 30
incre = 1
check_quit = False

def main():
    all_file_path = get_all_file_path(directory=input_directory, file_extension=['.mp4', '.avi', '.flv', '.mpg'])
    print(all_file_path)
    err_list = []
    for i in range(0, len(all_file_path)):
        # for i in range(0, 1):
        # try: # Tràn VRAM, tránh lỗi out chương trình
        for k in range(0, 1):
            print(all_file_path[i][FILENAME])
            summarize_video(video_path=all_file_path[i][FILEDIR],
                            output_path=output_directory + all_file_path[i][FILENAME] + dirsep,
                            filename=remove_extension(all_file_path[i][FILENAME]))
        # except:
        #     pass

Pause_check = False
def Pause():
    global Pause_check
    if Pause_check == False:
        Pause_check = True
    else:
        Pause_check = False
    pass

def forward():
    global incre
    incre = 1

def backward():
    global incre
    incre = -1

def increase_spd():
    global speed
    speed *= 0.9
    speed = max(1, round(speed))

def decrease_spd():
    global speed
    speed /= 0.9
    speed = min(900, round(speed))

def qt():
    global check_quit
    check_quit = True
    time.sleep(1.0)
    exit()

def New_Window():
    Window = Toplevel()
    Window.geometry("250x300")
    Window.title("Controller")
    PauseButton = Button(Window, height=2,
                         width=20,
                         text="Pause",
                         command=lambda: Pause())
    Forward = Button(Window, height=2,
                     width=20,
                     text="Forward",
                     command=lambda: forward())
    Backward = Button(Window, height=2,
                      width=20,
                      text="Backward",
                      command=lambda: backward())
    IncreaseSpeed = Button(Window, height=2,
                           width=20,
                           text="Speed++",
                           command=lambda: increase_spd())
    DecreaseSpeed = Button(Window, height=2,
                           width=20,
                           text="Speed--",
                           command=lambda: decrease_spd())
    CQ = Button(Window, height=2,
                width=20,
                text="Quit",
                command=lambda: qt())
    PauseButton.pack()
    Forward.pack()
    Backward.pack()
    IncreaseSpeed.pack()
    DecreaseSpeed.pack()
    CQ.pack()

def core(INPUT):
    temp = INPUT
    yt = YouTube(temp)
    mp4_files = yt.streams.filter(file_extension="mp4")
    mp4_369p_files = mp4_files.get_by_resolution("360p")
    mp4_369p_files.download(input_directory)
    try:
        main()
    except:
        pass
    folder = input_directory
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


root = Tk()
root.geometry("900x150")
root.title(" Visualizer ")

def Take_input():
    INPUT = inputtxt.get("1.0", "end-1c")
    threading.Thread(target=core, args=(INPUT, )).start()
    global check_quit

l = Label(text="Youtube link")
inputtxt = Text(root, height=3,
                width=100,
                bg="light yellow")

Display = Button(root, height=2,
                 width=20,
                 text="Visualize",
                 command=lambda: Take_input())

l.pack()
inputtxt.pack()
Display.pack()

mainloop()

# """READ YOUTUBE LINK"""
#
# temp = input()
# yt = YouTube(temp)
# mp4_files = yt.streams.filter(file_extension="mp4")
# mp4_369p_files = mp4_files.get_by_resolution("360p")
# mp4_369p_files.download(input_directory)
#
# """------------------------------------------------------------------------------------------------------------------"""


# try:
#     main()
# except:
#     pass
