__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.6'
__status__ = "Research"
__date__ = "1/12/2018"
__license__= "MIT License"


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as init

import random
import numpy as np

class LayerNorm(nn.Module):
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.gamma = nn.Parameter(torch.ones(features))
		self.beta = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.gamma * (x - mean) / (std + self.eps) + self.beta

class SelfAttention(nn.Module):

	def __init__(self, apperture=-1, ignore_itself=False, input_size=1024, output_size=1024):
		super(SelfAttention, self).__init__()

		self.apperture = apperture
		self.ignore_itself = ignore_itself

		self.m = input_size
		self.output_size = output_size

		self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
		self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
		self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
		self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)

		self.drop50 = nn.Dropout(0.5)



	def forward(self, x):
		n = x.shape[0]  # sequence length

		K = self.K(x)  # ENC (n x m) => (n x H) H= hidden size
		Q = self.Q(x)  # ENC (n x m) => (n x H) H= hidden size
		V = self.V(x)

		Q *= 0.06
		logits = torch.matmul(Q, K.transpose(1,0))

		if self.ignore_itself:
			# Zero the diagonal activations (a distance of each frame with itself)
			logits[torch.eye(n).byte()] = -float("Inf")

		if self.apperture > 0:
			# Set attention to zero to frames further than +/- apperture from the current one
			onesmask = torch.ones(n, n)
			trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
			logits[trimask == 1] = -float("Inf")

		att_weights_ = nn.functional.softmax(logits, dim=-1)
		weights = self.drop50(att_weights_)
		y = torch.matmul(V.transpose(1,0), weights).transpose(1,0)
		y = self.output_linear(y)

		return y, att_weights_



class VASNet(nn.Module):

	def __init__(self, m = 2048):
		super(VASNet, self).__init__()

		# self.m = 1024 # cnn features size
		self.m = m ## AN EDIT 2021.05.27 change features size to inceptionv3 2048
		# self.hidden_size = 1024  ## AN EDIT 2021.05.27 I don't know wtf does this do 

		self.att = SelfAttention(input_size=self.m, output_size=self.m)
		
		## AN EDIT
		#self.ka = nn.Linear(in_features=self.m, out_features=1024)
		#self.kb = nn.Linear(in_features=self.ka.out_features, out_features=1024)
		#self.kc = nn.Linear(in_features=self.kb.out_features, out_features=1024)
		#self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)

		self.ka = nn.Linear(in_features=self.m, out_features=self.m)
		self.kb = nn.Linear(in_features=self.ka.out_features, out_features=self.m)
		self.kc = nn.Linear(in_features=self.kb.out_features, out_features=self.m)
		self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)

		self.sig = nn.Sigmoid()
		self.relu = nn.ReLU()
		self.drop50 = nn.Dropout(0.5)
		self.softmax = nn.Softmax(dim=0)
		self.layer_norm_y = LayerNorm(self.m)
		self.layer_norm_ka = LayerNorm(self.ka.out_features)


	def forward(self, x, seq_len):

		m = x.shape[2] # Feature size

		# Place the video frames to the batch dimension to allow for batch arithm. operations.
		# Assumes input batch size = 1.
		x = x.view(-1, m)
		y, att_weights_ = self.att(x)

		y = y + x
		y = self.drop50(y)
		y = self.layer_norm_y(y)

		# Frame level importance score regression
		# Two layer NN
		y = self.ka(y)
		y = self.relu(y)
		y = self.drop50(y)
		y = self.layer_norm_ka(y)

		y = self.kd(y)
		y = self.sig(y)
		y = y.view(1, -1)

		return y, att_weights_



if __name__ == "__main__":
	pass

def weights_init(m):
	classname = m.__class__.__name__
	if classname == 'Linear':
		init.xavier_uniform_(m.weight, gain=np.sqrt(2.0))
		if m.bias is not None:
			init.constant_(m.bias, 0.1)

class AONet:

	def __init__(self):
		
		self.model = None

	def load_model(self, model_filename):
		self.model.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))
		return



	def initialize(self, cuda_device=None, f_len=1024):
		rnd_seed = 12345
		random.seed(rnd_seed)
		np.random.seed(rnd_seed)
		torch.manual_seed(rnd_seed)

		self.model = VASNet(m = f_len)
		self.model.eval()
		self.model.apply(weights_init)
		#print(self.model)

		cuda_device = cuda_device

		if cuda_device != None:
			print("Setting CUDA device: ",cuda_device)
			torch.cuda.set_device(cuda_device)
			torch.cuda.manual_seed(rnd_seed)

		if cuda_device != None:

			self.model.cuda()

		return


	def lookup_weights_file(self, data_path):
		dataset_type_str = '' if self.dataset_type == '' else self.dataset_type + '_'
		weights_filename = data_path + '/models/{}_{}splits_{}_*.tar.pth'.format(self.dataset_name, dataset_type_str, self.split_id)
		weights_filename = glob.glob(weights_filename)
		if len(weights_filename) == 0:
			print("Couldn't find model weights: ", weights_filename)
			return ''

		# Get the first weights filename in the dir
		weights_filename = weights_filename[0]
		splits_file = data_path + '/splits/{}_{}splits.json'.format(self.dataset_name, dataset_type_str)

		return weights_filename, splits_file



	def eval(self, features, results_filename=None):

		self.model.eval()
		
		
		with torch.no_grad():
			  seq = features
			  seq = torch.from_numpy(seq).unsqueeze(0)

			  # if self.hps.use_cuda:
			  #     seq = seq.float().cuda()

			  y, att_vec = self.model(seq, seq.shape[1])
			  summary = y[0].detach().cpu().numpy()
			  
		return summary



