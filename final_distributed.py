import os
import numpy as np
import random
import matplotlib.pyplot as plt
from operator import add
import matplotlib.cm as cm
import datetime
import math
from decimal import *
import time
from pyspark import SparkContext

sc = SparkContext()

eeta = 0.01
input_layer_neurons = 9        
hidden_layer_neurons = 4      #hidden layer
output_layer_neurons = 1       #output layer


def parse():

	dataset = np.loadtxt('/Users/chvinay/Desktop/dataset.txt',delimiter=',')
	data_train = np.loadtxt('/Users/chvinay/Desktop/dataset2.txt',delimiter=',')
	data = sc.parallelize(dataset)
	weights = [0.8,0.2]
	seed = 7777
	train_data, test_data = data.randomSplit(weights,seed) 

	return (train_data, data_train, test_data)


def act_sigmoid(val,derivative=False):

    if(derivative==True):
        return val*(1-val)
    return 1/(1+np.exp(-val))


def init_W(n1,n2):
	
	return np.zeros((n1,n2))


def rand_W(w):

	(n1,n2) = w.shape
	w = 2*np.random.random((n1,n2))-1
	return w


def ann_train(W, sample):

	w1, w2 = W
	x = sample[0:len(sample)-1]
	y = sample[-1]
	a1 = x

	z2 = np.dot(a1.T, w1)
	a2 = act_sigmoid(z2)

	z3 = np.dot(a2.T, w2)
	a3 = act_sigmoid(z3)
	delta3 = a3 - y
	delta2 = delta3*act_sigmoid(z2,True)
	Delta2 = delta2
	Delta1 = np.dot(w2.T,delta2) * act_sigmoid(a2,True)

	return np.array([Delta1, Delta2, a2, a3])

def eval_train(train_set, data_train, max_iter):
	Lambda = 9.01
	W0 = 2*np.random.random((input_layer_neurons,hidden_layer_neurons))-1
	W1 = 2*np.random.random((hidden_layer_neurons,output_layer_neurons))-1
	m = train_set.count()
	for iteration in range(max_iter):
		print 'iteration no: ', iteration
		eval_res = train_set.map(lambda x: ann_train((W0, W1), x))
		average_eval = eval_res.reduce(add) / train_set.count()
		dw1 = average_eval[0]
		dw2 = average_eval[1]
		l1 = average_eval[2]
		l2 = average_eval[3]

		for i in range(W0.shape[0]):
			for j in range(1, W0.shape[1]):
				W0[i, j] = W0[i, j] - (dw1[j] + (Lambda/m) * W0[i, j])

		for i in range(W1.shape[0]):
			for j in range(1, W1.shape[1]):
				W1[i,j] = W1[i, j] - (dw2[j] + (Lambda/m) * W1[i, j])

	return (W0,W1)


def ann_predict(W, inputs):
    W0,W1 = W

    z2 = np.dot(W0.T, inputs.T)
    a2 = act_sigmoid(z2)

    z3 = np.dot(W1.T, a2)
    a3 = act_sigmoid(z3)

    return a3



train_set, data_train, test_set = parse()

(W0,W1) = eval_train(train_set, data_train, 100)



n_test_set = test_set.count()

val_res = test_set.map(lambda x : 1 + np.argmax(ann_predict((W0,W1), x[0:len(x)-1]))).collect()
actual_res = test_set.map(lambda x : 1 + np.argmax(x[-1])).collect()

accurate = 0
for idx in range(n_test_set):
    if val_res[idx] == actual_res[idx]:
        accurate += 1
print("test set accuracy: %", accurate*100/n_test_set)