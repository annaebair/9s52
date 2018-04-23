import numpy as np
import csv
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing

def read_data(filename):
	data = []
	with open(filename) as f:
		reader = csv.reader(f)
		for row in reader:
			datarow = [float(x) for x in row]
			data.append(datarow)		
	return np.array(data)


def backprop(data, labels, hidden_units=20, learning_rate=0.1):
	weights_jk = 2 * np.random.rand(784, hidden_units) - 1
	weights_ij = 2 * np.random.rand(hidden_units, 10) - 1

	epochs = 50
	for epoch in range(epochs):
		data_order = np.arange(len(data))
		np.random.shuffle(data_order)
		num_correct = 0
		for index in data_order:

			data_point = data[index]
			target = labels[index]

			hidden_in = np.matmul(data_point, weights_jk)
			hidden_out = sigmoid(hidden_in)

			out_in = np.matmul(hidden_out, weights_ij)
			out_out = sigmoid(out_in)

			target_vec = vectorize_label(target)
			out_vec = winner(out_out)

			if np.array_equal(target_vec, out_vec):
				num_correct += 1
				continue

			error = target_vec - out_out

			delta_i = sigmoid_derivative(out_in) * error
			delta_wij = learning_rate * np.outer(delta_i, hidden_out)

			delta_j = sigmoid_derivative(hidden_in) * np.matmul(weights_ij, delta_i)
			delta_wjk = learning_rate * np.outer(delta_j, data_point)

			weights_ij += delta_wij.T
			weights_jk += delta_wjk.T

		print("Epoch ", epoch, ": ", num_correct)
	return weights_jk, weights_ij

	
def test_network(weights_jk, weights_ij, test_data, test_labels):
	num_correct = 0
	for index in range(len(test_data)):
		data_point = test_data[index]
		target = test_labels[index]

		hidden_in = np.matmul(data_point, weights_jk)
		hidden_out = sigmoid(hidden_in)

		out_in = np.matmul(hidden_out, weights_ij)
		out_out = sigmoid(out_in)

		target_vec = vectorize_label(target)
		out_vec = winner(out_out)

		if np.array_equal(target_vec, out_vec):
			num_correct += 1
			
	print(num_correct)


def winner(x):
	zeros = np.zeros(x.shape)
	zeros[np.argmax(x)] = 1
	return zeros


def vectorize_label(x):
	a = np.zeros(10)
	a[int(x)] = 1
	return a


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
	return sigmoid(x) * (1 - sigmoid(x))


if __name__ == '__main__':
	train_input = 'train_input.csv'
	train_output = 'train_output.csv'
	test_input = 'test_input.csv'
	test_output = 'test_output.csv'

	data = read_data(train_input)
	labels = read_data(train_output)
	test_data = read_data(test_input)
	test_labels = read_data(test_output)
	labels = np.squeeze(labels)
	labels[labels==[10]] = [0]
	test_labels = np.squeeze(test_labels)
	test_labels[test_labels==[10]] = [0]

	data1 = np.expand_dims(data[0], axis=1)
	weights_jk, weights_ij = backprop(np.array(data), labels)
	test_network(weights_jk, weights_ij, test_data, test_labels)

	
	### Code to show Images ###

	# for i in range(len(labels)):
	# 	# if labels[i] == [10]:
	# 		datarow = np.reshape(data[i], (28, 28)).T 
	# 		plt.figure()
	# 		plt.imshow(datarow, cmap='gray_r')
	# 		plt.title(labels[i])
	# 		plt.show()
