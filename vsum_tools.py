from ortools.algorithms import pywrapknapsack_solver
import os

import numpy as np

osolver = pywrapknapsack_solver.KnapsackSolver(
	# pywrapknapsack_solver.KnapsackSolver.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
	pywrapknapsack_solver.KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
	'test')

def knapsack_ortools(values, weights, items, capacity ):
	scale = 1000
	values = np.array(values)
	weights = np.array(weights)
	values = (values * scale).astype(np.int)
	weights = (weights).astype(np.int)
	capacity = capacity

	osolver.Init(values.tolist(), [weights.tolist()], [capacity])
	computed_value = osolver.Solve()
	packed_items = [x for x in range(0, len(weights))
					if osolver.BestSolutionContains(x)]

	return packed_items




import math


def generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15, method='knapsack'):
	"""Generate keyshot-based video summary i.e. a binary vector.
	Args:
	---------------------------------------------
	- ypred: predicted importance scores.
	- cps: change points, 2D matrix, each row contains a segment.
	- n_frames: original number of frames.
	- nfps: number of frames per segment.
	- positions: positions of subsampled frames in the original video.
	- proportion: length of video summary (compared to original video length).
	- method: defines how shots are selected, ['knapsack', 'rank'].
	"""
	n_segs = cps.shape[0]
	frame_scores = np.zeros((n_frames), dtype=np.float32)
	# if positions.dtype != int:
	#     positions = positions.astype(np.int32)
	if positions[-1] != n_frames:
		positions = np.concatenate([positions, [n_frames]])
	for i in range(len(positions) - 1):
		pos_left, pos_right = positions[i], positions[i+1]
		# print(pos_left, pos_right)
		if i == len(ypred):
			frame_scores[pos_left:pos_right] = 0
		else:
			frame_scores[pos_left:pos_right] = ypred[i]

	seg_score = []
	for seg_idx in range(n_segs):
		start, end = int(cps[seg_idx,0]), int(cps[seg_idx,1]+1)
		scores = frame_scores[start:end]
		seg_score.append(float(scores.mean()))

	limits = int(math.floor(n_frames * proportion))

	if method == 'knapsack':
		#picks = knapsack_dp(seg_score, nfps, n_segs, limits)
		picks = knapsack_ortools(seg_score, nfps, n_segs, limits)
	elif method == 'rank':
		order = np.argsort(seg_score)[::-1].tolist()
		picks = []
		total_len = 0
		for i in order:
			if total_len + nfps[i] < limits:
				picks.append(i)
				total_len += nfps[i]
	else:
		raise KeyError("Unknown method {}".format(method))

	summary = np.zeros((1), dtype=np.float32) # this element should be deleted
	for seg_idx in range(n_segs):
		nf = nfps[seg_idx]
		if seg_idx in picks:
			tmp = np.ones((nf), dtype=np.float32)
		else:
			tmp = np.zeros((nf), dtype=np.float32)
		summary = np.concatenate((summary, tmp))

	summary = np.delete(summary, 0) # delete the first element
	return summary


def evaluate_summary(machine_summary, user_summary, eval_metric='avg'):
	"""Compare machine summary with user summary (keyshot-based).
	Args:
	--------------------------------
	machine_summary and user_summary should be binary vectors of ndarray type.
	eval_metric = {'avg', 'max'}
	'avg' averages results of comparing multiple human summaries.
	'max' takes the maximum (best) out of multiple comparisons.
	"""
	machine_summary = machine_summary.astype(np.float32)
	user_summary = user_summary.astype(np.float32)
	n_users,n_frames = user_summary.shape

	# binarization
	machine_summary[machine_summary > 0] = 1
	user_summary[user_summary > 0] = 1

	if len(machine_summary) > n_frames:
		machine_summary = machine_summary[:n_frames]
	elif len(machine_summary) < n_frames:
		zero_padding = np.zeros((n_frames - len(machine_summary)))
		machine_summary = np.concatenate([machine_summary, zero_padding])

	f_scores = []
	prec_arr = []
	rec_arr = []

	for user_idx in range(n_users):
		gt_summary = user_summary[user_idx,:]
		overlap_duration = (machine_summary * gt_summary).sum()
		precision = overlap_duration / (machine_summary.sum() + 1e-8)
		recall = overlap_duration / (gt_summary.sum() + 1e-8)
		if precision == 0 and recall == 0:
			f_score = 0.
		else:
			f_score = (2 * precision * recall) / (precision + recall)
		f_scores.append(f_score)
		prec_arr.append(precision)
		rec_arr.append(recall)

	if eval_metric == 'avg':
		final_f_score = np.mean(f_scores)
		final_prec = np.mean(prec_arr)
		final_rec = np.mean(rec_arr)
	elif eval_metric == 'max':
		final_f_score = np.max(f_scores)
		max_idx = np.argmax(f_scores)
		final_prec = prec_arr[max_idx]
		final_rec = rec_arr[max_idx]
	
	return final_f_score, final_prec, final_rec


def evaluate_user_summaries(user_summary, eval_metric='avg'):
	"""Compare machine summary with user summary (keyshot-based).
	Args:
	--------------------------------
	machine_summary and user_summary should be binary vectors of ndarray type.
	eval_metric = {'avg', 'max'}
	'avg' averages results of comparing multiple human summaries.
	'max' takes the maximum (best) out of multiple comparisons.
	"""
	user_summary = user_summary.astype(np.float32)
	n_users, n_frames = user_summary.shape

	# binarization
	user_summary[user_summary > 0] = 1

	f_scores = []
	prec_arr = []
	rec_arr = []

	for user_idx in range(n_users):
		gt_summary = user_summary[user_idx, :]
		for other_user_idx in range(user_idx+1, n_users):
			other_gt_summary = user_summary[other_user_idx, :]
			overlap_duration = (other_gt_summary * gt_summary).sum()
			precision = overlap_duration / (other_gt_summary.sum() + 1e-8)
			recall = overlap_duration / (gt_summary.sum() + 1e-8)
			if precision == 0 and recall == 0:
				f_score = 0.
			else:
				f_score = (2 * precision * recall) / (precision + recall)
			f_scores.append(f_score)
			prec_arr.append(precision)
			rec_arr.append(recall)


	if eval_metric == 'avg':
		final_f_score = np.mean(f_scores)
		final_prec = np.mean(prec_arr)
		final_rec = np.mean(rec_arr)
	elif eval_metric == 'max':
		final_f_score = np.max(f_scores)
		max_idx = np.argmax(f_scores)
		final_prec = prec_arr[max_idx]
		final_rec = rec_arr[max_idx]

	return final_f_score, final_prec, final_rec


def generate_summary_video(video_path, sum_video_path, summary):
	import cv2
	# os.remove(f"{video_path}/{sum_video_path}")

	# %rm -rf $sum_video_path
	# %mkdir $sum_video_path

	video = cv2.VideoCapture(video_path)
	fps = video.get(cv2.CAP_PROP_FPS)
	frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
	size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
	size_param = "%dx%d"%size 
	padding = lambda i: '0'*(6-len(str(i))) + str(i)

	print(fps, frameCount, size)

	# videoWriter = cv2.VideoWriter('trans.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, size)  

	success, frame = video.read()  
	index = 0
	choosen = 0

	
	while success :  
		if summary[index] == 1:
			cv2.imwrite(sum_video_path + "\\selected_frame-%s.png"%(padding(choosen)), frame)
			# cv2.imwrite("current" + "\\save_frame\\selected_frame_" + str(index) + ".png", frame)
			# print(frame.shape)
			# print(frame)
			# print(index, end=' ', )
			choosen += 1
		success, frame = video.read()
		index += 1

	# print(f"Generating sum_video_path)
	sum_video_path = "\\" + sum_video_path
	os.system(f"cd {sum_video_path} ; ffmpeg -f image2 -framerate {fps} -i selected_frame-%06d.png -s {size_param} -c:v h264 {sum_video_path}\\summary.mp4")
	
	#Remove selected frame after generate summary
	os.system(f"cd {sum_video_path} ; rm selected_frame*.png")


# https://www.tensorflow.org/api_docs/python/tf/keras/applications
# https://keras.io/api/applications/

def model_picker(name, pooling_type):
	
	from tensorflow.keras.applications import imagenet_utils
	preprocess = imagenet_utils.preprocess_input

	if (name == 'vgg16'):
		from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input 
		targetsize = (224, 224, 3)
		model = VGG16(weights='imagenet',
					  include_top=False,
					  input_shape=targetsize,
					  pooling=pooling_type)
		preprocess = preprocess_input
	elif (name == 'vgg19'):
		from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
		targetsize = (224, 224, 3)
		model = VGG19(weights='imagenet',
					  include_top=False,
					  input_shape=targetsize,
					  pooling=pooling_type)
		preprocess = preprocess_input
	elif (name == 'mobilenet'):
		from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
		targetsize = (224, 224, 3)
		model = MobileNet(weights='imagenet',
						  include_top=False,
						  input_shape=targetsize,
						  pooling=pooling_type,
						  depth_multiplier=1,
						  alpha=1)
		preprocess = preprocess_input
	elif (name == 'inceptionv3'):
		from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
		targetsize = (299, 299, 3)
		model = InceptionV3(weights='imagenet',
							include_top=False,
							input_shape=targetsize,
							pooling=pooling_type)
		preprocess = preprocess_input
	elif (name == 'inceptionresnetv2'):
		from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
		targetsize = (299, 299, 3)
		model = InceptionResNetV2(weights='imagenet',
							include_top=False,
							input_shape=targetsize,
							pooling=pooling_type)
		preprocess = preprocess_input
	elif (name == 'resnet50'):
		from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
		targetsize = (224, 224, 3)
		model = ResNet50(weights='imagenet',
						 include_top=False,
						 input_shape=targetsize,
						 pooling=pooling_type)
		preprocess = preprocess_input
	elif (name == 'xception'):
		from tensorflow.keras.applications.xception import Xception, preprocess_input
		targetsize = (224, 224, 3)
		model = Xception(weights='imagenet',
						 include_top=False,
						 input_shape=targetsize,
						 pooling=pooling_type)
		preprocess = preprocess_input
	elif (name == 'nasnetlarge'):
		from tensorflow.keras.applications.nasnet import NASNetLarge, preprocess_input
		targetsize = (331, 331, 3)
		model = NASNetLarge(weights='imagenet',
						 include_top=False,
						 input_shape=targetsize,
						 pooling=pooling_type)
		preprocess = preprocess_input
	elif (name == 'efficientnetb3'):
		from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input
		targetsize = (224, 224, 3)
		model = EfficientNetB3(weights='imagenet',
						 include_top=False,
						 input_shape=targetsize,
						 pooling=pooling_type)
		preprocess = preprocess_input
	elif (name == 'efficientnetb5'):
		from tensorflow.keras.applications.efficientnet import EfficientNetB5, preprocess_input
		targetsize = (240, 240, 3)
		model = EfficientNetB5(weights='imagenet',
						 include_top=False,
						 input_shape=targetsize,
						 pooling=pooling_type)
		preprocess = preprocess_input
	elif (name == 'efficientnetb7'):
		from tensorflow.keras.applications.efficientnet import EfficientNetB7, preprocess_input
		targetsize = (240, 240, 3)
		model = EfficientNetB7(weights='imagenet',
						 include_top=False,
						 input_shape=targetsize,
						 pooling=pooling_type)
		preprocess = preprocess_input
	else:
		print("Specified model not available")
		
	target_size = (targetsize[0], targetsize[1])
	return model, preprocess, target_size

import cv2

def extract_features4video(model_func, preprocess_func, target_size, picks, video_file):
	
	from tensorflow.keras.preprocessing import image
	import numpy as np

	# initialize the video stream, pointer to output video file, and frame dimensions
	vs = cv2.VideoCapture(video_file)
	writer = None
	(W, H) = (None, None)
	# loop over frames from the video file stream

	features = []
	pick_frames = []

	n_frames = -1

	total = len(picks)
	count = 0

	selected_frame = picks[count]

	while True:
		# read the next frame from the file
		(grabbed, frame) = vs.read()
		# if the frame was not grabbed, then we have reached the end of the stream
		if not grabbed:
			break

		n_frames += 1

		#if (n_frames in picks): // banana code - làm chậm 10 lần
		if( selected_frame == n_frames):
			count += 1
			if(count < total):
				selected_frame = picks[count]
			if(count % 50 == 0):
				print("-- Processing frame ", count, "/", total, ": ", n_frames)
			# convert it from BGR to RGB
			# ordering, resize the frame to a fixed target_size, and then
			# perform mean subtraction
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame = cv2.resize(frame, target_size).astype("float32")
			img_array = image.img_to_array(frame)
			expanded_img_array = np.expand_dims(img_array, axis=0)
			preprocessed_img = preprocess_func(expanded_img_array)
			pick_frames.append(preprocessed_img)
			
			# print(n_frames)
			# debug only
			#if(count>10):
			#  break

	pick_frames = np.vstack(pick_frames)
	print("pick frames ", len(pick_frames))
	# xử lý theo batch
	print("-- Extracting features: ...")
	features = model_func.predict(pick_frames, batch_size = 32)
	print("-- Done. Feature size: ", features.shape)

	return features


def extract_frame_and_features4video(model_func, preprocess_func, target_size, picks, video_file):
	
	from tensorflow.keras.preprocessing import image
	import numpy as np

	# initialize the video stream, pointer to output video file, and frame dimensions
	vs = cv2.VideoCapture(video_file)
	writer = None
	(W, H) = (None, None)
	# loop over frames from the video file stream

	features = []
	pick_frames = []

	n_frames = -1

	total = len(picks)
	count = 0

	import time
	start = time.time()

	selected_frame = picks[count]
	frame_before_converting = []

	while True:
		# read the next frame from the file
		(grabbed, frame) = vs.read()
		# if the frame was not grabbed, then we have reached the end of the stream
		if not grabbed:
			break

		n_frames += 1

		#if (n_frames in picks): // banana code - làm chậm 10 lần
		if( selected_frame == n_frames):
			count += 1
			if(count < total):
				selected_frame = picks[count]
			if(count % 50 == 0):
				print("-- Processing frame ", count, "/", total, ": ", n_frames)
			# convert it from BGR to RGB
			# ordering, resize the frame to a fixed target_size, and then
			# perform mean subtraction
			frame_before_converting.append(frame)
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame = cv2.resize(frame, target_size).astype("float32")
			img_array = image.img_to_array(frame)
			expanded_img_array = np.expand_dims(img_array, axis=0)
			preprocessed_img = preprocess_func(expanded_img_array)
			pick_frames.append(preprocessed_img)
			
		 
	extract_frame_time = time.time()

	print(f"Extracting frame linearly using opencv took {extract_frame_time - start:.2f} seconds")

	pick_frames = np.vstack(pick_frames)
	print("pick frames ", len(pick_frames))
	# xử lý theo batch
	print("-- Extracting features: ...")
	features = model_func.predict(pick_frames, batch_size = 32)
	print("-- Done. Feature size: ", features.shape)

	extract_features_time = time.time()
	print(f"Extracting frame feature using {model_func} took  {extract_features_time - extract_frame_time:.2f} seconds")


	return frame_before_converting, features


def save_frame( picks, video_file, dest_dir):
	
	pick_frames = []

	n_frames = -1

	vs = cv2.VideoCapture(video_file)


	total = len(picks)
	count = 0

	import time
	start = time.time()

	selected_frame = picks[count]
	frame_before_converting = []

	while True:
		# read the next frame from the file
		(grabbed, frame) = vs.read()
		# if the frame was not grabbed, then we have reached the end of the stream
		if not grabbed:
			break

		n_frames += 1

		#if (n_frames in picks): // banana code - làm chậm 10 lần
		if( selected_frame == n_frames):
			count += 1
			if(count < total):
				selected_frame = picks[count]
			if(count % 50 == 0):
				print("-- Processing frame ", count, "/", total, ": ", n_frames)
			# convert it from BGR to RGB
			# ordering, resize the frame to a fixed target_size, and then
			# perform mean subtraction

			cv2.imwrite(f"{dest_dir}/frame_{n_frames:06d}.png", frame)
		
def load_saved_frame(dest_dir):
	import glob
	files = sorted( glob.glob(f"{dest_dir}/frame_*.png") )

	frames = []
	for i in files:
		frames.append(cv2.imread(i))

	return frames