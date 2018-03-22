import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import csv
import math


def get_data(datafile):
	all_data = []
	all_labels = []
	setosa_data = []
	versicolor_data = []
	virginica_data = []
	non_setosa_labels = []
	file = open(datafile, 'r');
	reader = csv.reader(file)
	for row in reader:
		no_name = row[:-1]
		flower_type = row[-1]
		float_row = [float(i) for i in no_name]
		all_data.append(float_row)
		if flower_type == "Iris-setosa":
			all_labels.append(-1)
			setosa_data.append(float_row)
		elif flower_type == "Iris-versicolor":
			all_labels.append(1)
			versicolor_data.append(float_row)
			non_setosa_labels.append(-1)
		else:
			all_labels.append(1)
			virginica_data.append(float_row)
			non_setosa_labels.append(1)
	return np.array(all_data), np.array(all_labels), np.array(setosa_data), np.array(versicolor_data), np.array(virginica_data), np.array(non_setosa_labels)

def perceptron_iteration(weight_matrix, num_data_points, data, labels, learning_rate):
	correct = 0
	for i in range(num_data_points):
		point = data[i]
		computed_output = np.matmul(point, weight_matrix)
		if computed_output > 0:
			observed = 1
		else:
			observed = -1
		actual = labels[i]
		if observed != actual:
			update = learning_rate * (actual - observed) * point
			weight_matrix += update
			total = sum(weight_matrix)
			weight_matrix /= total
		else:
			correct += 1
	return weight_matrix, correct


def perceptron(data, labels, learning_rate):
	num_data_points, num_attributes = data.shape
	weight_matrix = np.random.rand(num_attributes)
	iterations = 10
	for i in range(iterations):
		weight_matrix, num_correct = perceptron_iteration(weight_matrix, num_data_points, data, labels, learning_rate)
		print(f"Iteration {i+1}: {num_correct} correctly classified")


def sanger(data, learning_rate):
	centered_data = StandardScaler().fit_transform(data)
	num_data_points, num_attributes = centered_data.shape
	weight_matrix = np.random.rand(num_attributes, num_attributes)
	print("original weight matrix:\n")
	print(weight_matrix)
	# print(centered_data)
	output = np.matmul(centered_data, weight_matrix)
	for it in range(150000):
		d = np.random.randint(0, 150)
		x = centered_data[d] # one input instance
		v = output[d] # one output instance
		for j in range(len(x)):
			xj = x[j] # single input node value for one data point
			for i in range(len(v)):
				vi = v[i] # single output node associated with this data point
				sum_term = 0
				for k in range(i+1):
					sum_term += v[k] * weight_matrix[j][k]
				delta_w_ij = learning_rate * vi * (xj - sum_term)
				weight_matrix[j][i] += delta_w_ij
		output = np.matmul(centered_data, weight_matrix)
	print("final weight matrix: \n")
	print(weight_matrix)
	covariance_mat = np.cov(np.transpose(centered_data))
	print("data covariance matrix: \n")
	print(covariance_mat)
	eigenval, eigenvec = np.linalg.eig(covariance_mat)
	print("eigenvectors: \n")
	print(eigenvec)
	print("covariance between weight matrix: \n")
	weight_cov = np.cov(weight_matrix)
	print(weight_cov)
	pca = PCA().fit_transform(covariance_mat)
	print("pca of matrix covariance: \n")
	print(pca)
	diag_sum = 0
	#okay to use sklearn to find variance explained
	for i in range(4):
		diag_sum += abs(eigenvec[i][i])
	for i in range(4):
		proportion = eigenvec[i][i]/diag_sum
		print(f"{i}: {proportion}")

	# print("pca of weight covariance: \n")
	# print(PCA().fit_transform(weight_cov))

	# columns should be directions of PCs; col 1 is the largest PC

def adaline(data, labels, learning_rate):
	num_data_points, num_attributes = data.shape
	weight_matrix = np.random.rand(4)
	iterations = 500
	for i in range(num_data_points):
		inp = data[i]
		actual = labels[i]
		observed = np.matmul(inp, weight_matrix)
		weight_update = learning_rate * (actual - observed) * inp
		weight_matrix += weight_update
		weight_matrix /= sum(weight_matrix)
	print(weight_matrix)
	adaline_error(data, weight_matrix, labels)

def adaline_error(data, weight_matrix, labels):
	out = np.matmul(data, weight_matrix)
	error = 0.5 * sum((labels - out)**2)
	print(error)

def sofm():
	learning_rate = 0.05
	input_data = np.random.uniform(size=(5000, 2))
	plt.plot(input_data[:, 0], input_data[:, 1], 'ko', markersize=2)

	grid = np.random.rand(100, 2)
	print(grid)
	original_grid = grid.copy()
	# plt.plot(original_grid[:, 0], original_grid[:, 1], 'ro')
	iterations = 10
	for it in range(iterations):
		print(f"Iteration {it}")
		for point in input_data:
			#calculate distance to each grid point
			BMU = None 
			min_distance = np.inf
			BMU_index = None
			for i in range(len(grid)):
				node = grid[i]
				dist = euclidian_distance(point, node)
				if dist < min_distance:
					min_distance = dist
					BMU = node
					BMU_index = i
			# update weight of BMU and surrounding neurons
			radius = 0.05/(it+1)
			# print(f"radius: {radius}")
			indices = []
			for j in range(len(grid)):
				node = grid[j]
				dist = euclidian_distance(BMU, node)
				if  dist < radius:
					# indices.append(j)
					n = grid[j]
					gaussian = 1/(math.sqrt(2*math.pi) * math.exp(-0.5 * dist))
					# print(gaussian)
					new_n = n + learning_rate * gaussian * (point - n)
					grid[j] = new_n
			# for ind in indices:
			# 	n = grid[ind]
			# 	new_n = n + learning_rate * (point - n)
			# 	grid[ind] = new_n
	
	# grid = np.sort(grid, axis=1)
	# print(grid)
	plt.plot(grid[:, 0], grid[:, 1], 'ro')
	plt.show()


def euclidian_distance(x, y):
	return math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)

def visualize(x, y, setosa, versicolor, virginica):
	dim_1_setosa = [row[0] for row in setosa]
	dim_2_setosa = [row[1] for row in setosa]
	dim_3_setosa = [row[2] for row in setosa]
	dim_4_setosa = [row[3] for row in setosa]
	dim_1_versicolor = [row[0] for row in versicolor]
	dim_2_versicolor = [row[1] for row in versicolor]
	dim_3_versicolor = [row[2] for row in versicolor]
	dim_4_versicolor = [row[3] for row in versicolor]
	dim_1_virginica = [row[0] for row in virginica]
	dim_2_virginica = [row[1] for row in virginica]
	dim_3_virginica = [row[2] for row in virginica]
	dim_4_virginica = [row[3] for row in virginica]

	fig = plt.figure()
	plt.subplot(221)
	setosa_points = plt.scatter(dim_1_setosa, dim_2_setosa, c='b')
	versicolor_points = plt.scatter(dim_1_versicolor, dim_2_versicolor, c='g')
	virginica_points = plt.scatter(dim_1_virginica, dim_2_virginica, c='m')
	plt.xlabel("1")
	plt.ylabel("2")
	plt.subplot(222)
	plt.scatter(dim_2_setosa, dim_3_setosa, c='b')
	plt.scatter(dim_2_versicolor, dim_3_versicolor, c='g')
	plt.scatter(dim_2_virginica, dim_3_virginica, c='m')
	plt.xlabel("2")
	plt.ylabel("3")
	plt.subplot(223)
	plt.scatter(dim_3_setosa, dim_4_setosa, c='b')
	plt.scatter(dim_3_versicolor, dim_4_versicolor, c='g')
	plt.scatter(dim_3_virginica, dim_4_virginica, c='m')
	plt.xlabel("3")
	plt.ylabel("4")
	plt.subplot(224)
	plt.scatter(dim_4_setosa, dim_1_setosa, c='b')
	plt.scatter(dim_4_versicolor, dim_1_versicolor, c='g')
	plt.scatter(dim_4_virginica, dim_1_virginica, c='m')
	plt.xlabel("4")
	plt.ylabel("1")
	fig.legend((setosa_points, versicolor_points, virginica_points), ("setosa", "versicolor", "virginica"))
	plt.show()

if __name__ == "__main__":

	datafile = "data.txt"
	x, y, setosa, versicolor, virginica, non_setosa_labels = get_data(datafile)
	learning_rate = 0.00001
	non_setosa_data = np.concatenate((versicolor, virginica), axis=0)
	# print(non_setosa_data)
	# perceptron(x, y, learning_rate)
	# visualize(x, y, setosa, versicolor, virginica)
	# setosa is linearly separable from the other two
	# sanger(x, learning_rate)
	# adaline(non_setosa_data, non_setosa_labels, learning_rate)
	sofm()
	# pca = PCA(n_components=2).fit_transform(x)
	# print(pca)

