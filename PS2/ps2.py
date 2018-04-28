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


def backprop(data, labels, epochs, test_data, test_labels, hidden_units=20, learning_rate=0.1, momentum=None):
	
	# Random initialization of weight matrices, centered at 0
	weights_jk = 2 * np.random.rand(784, hidden_units) - 1
	weights_ij = 2 * np.random.rand(hidden_units, 10) - 1

	train_correct = []
	test_correct = []

	for epoch in range(epochs):

		# Randomly shuffle order in which to choose data points
		data_order = np.arange(len(data))
		np.random.shuffle(data_order)
		num_correct = 0

		for index in data_order:

			# Get a random data point and corresponding target label
			data_point = data[index]
			target = labels[index]

			# Input to and output from hidden layer
			hidden_in = np.matmul(data_point, weights_jk)
			hidden_out = sigmoid(hidden_in)

			# Input to and output from output layer
			out_in = np.matmul(hidden_out, weights_ij)
			out_out = sigmoid(out_in)

			# Convert target label to a binary vector
			target_vec = vectorize_label(target)

			# Convert output vector to a winner take all binary vector
			out_vec = winner(out_out)

			# Correct classification if output binary vector matches target binary vector
			if np.array_equal(target_vec, out_vec):
				num_correct += 1
				continue

			# If incorrect classification, calculate the error
			error = target_vec - out_out

			# Calculate weight update for weight matrix W (hidden to output)
			delta_i = sigmoid_derivative(out_in) * error
			delta_wij = learning_rate * np.outer(delta_i, hidden_out)

			# Calculate weight update for weight matrix w (input to hidden)
			delta_j = sigmoid_derivative(hidden_in) * np.matmul(weights_ij, delta_i)
			delta_wjk = learning_rate * np.outer(delta_j, data_point)

			if momentum:
				# Update rule including momentum term
				weights_ij = momentum * weights_ij + delta_wij.T
				weights_jk = momentum * weights_jk + delta_wjk.T
			else:
				# Update rule if no momentum
				weights_ij += delta_wij.T
				weights_jk += delta_wjk.T

		train_correct.append(1-num_correct/3000)
		test_performance = test_network(weights_jk, weights_ij, test_data, test_labels)
		test_correct.append(1-test_performance/3000)
		print("Epoch ", epoch, ' | Train: ',num_correct/3000, ' | Test: ', test_performance/3000)

	return train_correct, test_correct

	
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
			
	return num_correct


def winner(x):
	# Converts an output vector to a binary winner-take-all vector of the same shape
	zeros = np.zeros(x.shape)
	zeros[np.argmax(x)] = 1
	return zeros


def vectorize_label(x):
	# Converts an integer label to a binary 10x1 vector
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

	epochs = 100
	hidden_units = 20
	learning_rate = 0.1
	momentum=0.9

	# Run backprop
	train, test = backprop(data, labels, epochs, test_data, test_labels, hidden_units, learning_rate)

	# Plot train and test performance
	plt.plot(train, label="Train Performance")
	plt.plot(test, label="Test Performance")
	plt.xlabel("Iterations")
	plt.ylabel("Error Rate")
	plt.title("Train and Test Performance on MNIST")
	plt.legend()
	# plt.show()

	
	# Code to show MNIST Images
	
	for i in range(len(labels)):
		datarow = np.reshape(data[i], (28, 28)).T 
		plt.figure()
		plt.imshow(datarow, cmap='gray_r')
		plt.title(labels[i])
		plt.show()
