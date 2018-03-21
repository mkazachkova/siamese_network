#import matplotlib.pyplot as plt
import torch
from torch import autograd
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torch.nn as nn
import random
import sys


class custome_Dataset(Dataset):

	def __init__(self,images,train,aug):
		self.images = images #folder with images
		self.train = train #text file with name of images
		self.aug = aug #if augmenting

	#return image pair and label; make images correct size
	def __getitem__(self, index):
		line = self.train[index].split() #get name of 2 images and label
		img1 = cv2.imread(self.images + line[0] , 1)
		img2 = cv2.imread(self.images + line[1] , 1)
		label = int(line[2])
		
		img1 = cv2.resize(img1,(128,128))
		img2 = cv2.resize(img2,(128,128))

		height1 = img1.shape[0]
		width1 = img1.shape[1]
		height2 = img2.shape[0]
		width2 = img2.shape[1]
		center1 = (width1 / 2, height1 / 2)
		center2 = (width2 / 2, height2 / 2)



		"""Start data augmentation"""
		if self.aug:
			if random.random() < .7: #apply transformation
				img1 = cv2.flip(img1,1) #mirror

				rotation1 = random.uniform(-30,30)
				Mat1 = cv2.getRotationMatrix2D(center1, rotation1, 1.0)
				img1 = cv2.warpAffine(img1, Mat1, (width1, height1)) #rotate

				transx_1 = random.randint(-10,10)
				transy_1 = random.randint(-10,10)
				M1 = np.float32([[1,0,transx_1],[0,1,transy_1]])
				img1 = cv2.warpAffine(img1,M1,(width1,height1)) #translate
				

				resize1 = random.uniform(.7,1.3)
				img1 = cv2.resize(img1, (0,0), fx=resize1, fy=resize1) #rescale
				img1 = cv2.resize(img1,(128,128))
		
			if random.random() < .7: #apply transformation
				rotation2 = random.uniform(-30,30)
				Mat2 = cv2.getRotationMatrix2D(center2, rotation2, 1.0)
				img2 = cv2.warpAffine(img2, Mat2, (width2, height2)) #rotate

				transx_2 = random.randint(-10,10)
				transy_2 = random.randint(-10,10)
				M2 = np.float32([[1,0,transx_2],[0,1,transy_2]])
				img2 = cv2.warpAffine(img2,M2,(width2,height2)) #translate

				resize2 = random.uniform(.7,1.3)
				img2 = cv2.resize(img2, (0,0), fx=resize2, fy=resize2) #rescale

				img2 = cv2.resize(img2,(128,128))	



		return img1,img2,label

	def __len__(self):
		return len(self.train)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class Siamese_Network(nn.Module):
	def __init__(self):
		super(Siamese_Network, self).__init__()
		#define our neural network
		self.first = nn.Sequential(
		   nn.Conv2d(3, 64, kernel_size=5, stride=(1,1),padding=2),
		   nn.ReLU(inplace=True), #relu in place
		   nn.BatchNorm2d(64), #normalization 64 elements
		   nn.MaxPool2d(2,stride=(2,2)), #max pooling layer
		   nn.Conv2d(64, 128, kernel_size=5, stride=(1,1),padding=2),
		   nn.ReLU(inplace=True), #relu in place
		   nn.BatchNorm2d(128), #normalization 128 elements
		   nn.MaxPool2d(2,stride=(2,2)), #max pooling layer
		   nn.Conv2d(128, 256, kernel_size=3, stride=(1,1),padding=1),
		   nn.ReLU(inplace=True), #relu in place
		   nn.BatchNorm2d(256), #normalization 128 elements
		   nn.MaxPool2d(2,stride=(2,2)), #max pooling layer
		   nn.Conv2d(256, 512, kernel_size=3, stride=(1,1),padding=1),
		   nn.ReLU(inplace=True), #relu in place
		   nn.BatchNorm2d(512), #normalization 128 elements
		   Flatten(), #flatten to 1d 
		   nn.Linear(131072,1024), #fully connected
		   nn.ReLU(inplace=True), #relu in place
		   nn.BatchNorm2d(1024) #normalization 128 elements

		)

		self.second = nn.Sequential(
			nn.Linear(2048,1), #fully connected	
			nn.Sigmoid()
		)

	def forward(self, img1,img2):
		f1 = self.first(img1)
		f2 = self.first(img2) #pass both through network
		f12 = torch.cat((f1,f2),1)
		return self.second(f12)



"""
passed = sys.argv[1].split()
if passed[0] == '--load':
	test(passed[1])
	
else if passed[0] == '--save'
	train(passed[1])
else:
	print 'unknown or no argument'
	exit()
"""

def train(name):

	C = Siamese_Network().cuda()
	N = 32
	#loading data
	f = open('train.txt', 'r').readlines()
	train = []
	for line in f:
		train.append(line)

	data = custome_Dataset('lfw/',train,True)
	loader = DataLoader(data,shuffle=True,num_workers=2,batch_size=N,drop_last=True)
	loss_fn = nn.BCELoss()
	learning_rate = .0001
	optimizer = torch.optim.Adam(C.parameters(), lr=learning_rate)
	losses = []
	loss = None

	for epoch in range(30):

		for i,data in enumerate(loader,0):
			#optimizer.zero_grad()
			img1tt, img2tt , label = data
			img1 = img1tt.resize_((N,3,128,128))

			img2 = img2tt.resize_((N,3,128,128))
			label = label.resize_((N,1)) #make label variable
			img1, img2 , label = Variable(img1).float().cuda(), Variable(img2).float().cuda() , Variable(label).float().cuda()
			val = C(img1,img2)
			optimizer.zero_grad()
			loss = loss_fn(val, label)
			#print loss
			loss.backward(retain_graph=True)
			optimizer.step()
		print 'epoch: ',
		print epoch
		print 'loss: ',
		print loss.cpu().data.numpy()[0]
	torch.save(C.state_dict(), name)

def test(name):
	C = Siamese_Network().cuda()
	N = 32
	C.load_state_dict(torch.load(name))

	f = open('train.txt', 'r').readlines()
	train = []
	for line in f:
		train.append(line)

	data_t = custome_Dataset('lfw/',train,False)
	loader_t = DataLoader(data_t,shuffle=False,num_workers=2,batch_size=N,drop_last=True)

	count = 0
	correct = 0
	C.eval() #because otherwise things are bad
	for i, data in enumerate(loader_t,0):
		count+=N #batch size of 32
		img1, img2 , label = data
		img1 = img1.resize_((N,3,128,128))
		img2 = img2.resize_((N,3,128,128))
		label = label.resize_((N,1))
		img1, img2 = Variable(img1).float().cuda(), Variable(img2).cuda().float()
		#label = label.cuda()
		val = C(img1,img2).float()
		y_hat = None
		y_hat_round = []
		for element in val.cpu().data.numpy():
			if element[0] > .5:
				y_hat_round.append([1])
			else:
				y_hat_round.append([0])
		correct += np.count_nonzero(label.numpy() == np.array(y_hat_round))

	print "Train accuracy is ",(correct / float(count))



	"""Test train acuracy"""
	test_file = open('test.txt', 'r').readlines()
	test = []
	for line in test_file:
		test.append(line)

	data_test = custome_Dataset('lfw/',test,False)
	test_loader = DataLoader(data_test,shuffle=False,batch_size=N,drop_last=True)
	count = 0
	correct = 0

	for i, data_test in enumerate(test_loader,0):
		count+=N #batch size of 32
		img1, img2 , label = data_test
		img1 = img1.resize_((N,3,128,128))
		img2 = img2.resize_((N,3,128,128))
		label = label.resize_((N,1))
		img1, img2 = Variable(img1).cuda().float(), Variable(img2).cuda().float()
		val = C(img1,img2).float()
		y_hat = None
		y_hat_round = []
		for element in val.cpu().data.numpy():
			if element[0] > .5:
				y_hat_round.append([1])
			else:
				y_hat_round.append([0])
		correct += np.count_nonzero(label.numpy() == np.array(y_hat_round))


	print "Test accuracy is ",(correct / float(count))


passed = sys.argv[1]
if passed == '--load':
	test(sys.argv[2])
	
elif passed == '--save':
	train(sys.argv[2])
else:
	print 'unknown or no argument'
	exit()

