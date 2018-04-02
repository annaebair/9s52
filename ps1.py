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
	iterations = 100
	proportion_correct = []
	for i in range(iterations):
		weight_matrix, num_correct = perceptron_iteration(weight_matrix, num_data_points, data, labels, learning_rate)
		proportion_correct.append(num_correct/150)
		print(f"Iteration {i+1}: {num_correct} correctly classified")
	return proportion_correct
	

def plot_perceptrons(data, labels, learning_rates):
	all_proportions_correct = []
	for r in learning_rates:
		proportion_correct = perceptron(data, labels, r)
		all_proportions_correct.append(proportion_correct)

	plt.subplot(311)
	plt.plot(np.arange(100), all_proportions_correct[0])
	plt.title("Rate of Convergence with Learning Rate 0.01")
	plt.xlabel("Iterations")
	plt.ylabel("Proportion Correct")
	plt.subplot(312)
	plt.plot(np.arange(100), all_proportions_correct[1])
	plt.title("Rate of Convergence with Learning Rate 0.0001")
	plt.xlabel("Iterations")
	plt.ylabel("Proportion Correct")
	plt.subplot(313)
	plt.plot(np.arange(100), all_proportions_correct[2])
	plt.title("Rate of Convergence with Learning Rate 0.00001")
	plt.xlabel("Iterations")
	plt.ylabel("Proportion Correct")
	plt.show()


def sanger(data, learning_rate):
	centered_data = StandardScaler().fit_transform(data)
	num_data_points, num_attributes = centered_data.shape
	weight_matrix = np.random.rand(num_attributes, num_attributes)
	output = np.matmul(centered_data, weight_matrix)
	for it in range(300000):
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
	eigenval, eigenvec = np.linalg.eig(covariance_mat)
	total_eigs = sum(eigenval)
	reverse_sorted_eigenvals = sorted(eigenval, reverse=True)
	amt_of_var = [eig/total_eigs for eig in reverse_sorted_eigenvals]
	cumulative_variance = sorted(np.cumsum(amt_of_var))

	plt.plot(([1,2,3,4]), cumulative_variance)
	plt.ylim(0,1.1)
	plt.ylabel("Amount of Variance explained")
	plt.xlabel("Principal Component")
	plt.xticks([1,2,3,4], ("PC1", "PC2", "PC3", "PC4"))
	plt.title("Cumulative Variance Explained by each Principal Component")
	plt.show()

	covariance_mat = np.cov(np.transpose(centered_data))
	eigenval, eigenvec = np.linalg.eig(covariance_mat)


def reduce_dims(data, labels, non_setosa_labels, plot=True):
	centered_data = StandardScaler().fit_transform(data)
	covariance_mat = np.cov(np.transpose(centered_data))
	eigenval, eigenvec = np.linalg.eig(covariance_mat)
	eigs = [(eigenval[i], eigenvec[:, i]) for i in range(4)]
	sorted_eigs = sorted(eigs, reverse=True)
	pcs = [sorted_eigs[i][1] for i in range(2)]
	conversion = np.array(pcs)
	projection = centered_data.dot(np.transpose(conversion))
	setosa = []
	versicolor = []
	virginica = []
	non_set = []
	for i in range(150):
		if labels[i] == -1:
			setosa.append(projection[i])
		else:
			non_set.append(projection[i])

	for i in range(100):
		if non_setosa_labels[i] == -1:
				versicolor.append(non_set[i])
		else:
			virginica.append(non_set[i])
	setosa = np.array(setosa)
	virginica = np.array(virginica)
	versicolor = np.array(versicolor)
	if plot:
		plt.figure()
		setosa_points = plt.scatter(setosa[:,0], setosa[:,1], c='b')
		versicolor_points = plt.scatter(versicolor[:,0], versicolor[:,1], c='g')
		virginica_points = plt.scatter(virginica[:,0], virginica[:,1], c='m')
		plt.title("Iris Dataset projected onto first two PCs")
		plt.legend((setosa_points, versicolor_points, virginica_points), ("Setosa", "Versicolor", "Virginica"))
		plt.show()
	return projection


def adaline_error(data, weight_matrix, labels):
	out = np.matmul(data, weight_matrix)
	error = 0.5 * sum((labels - out)**2)
	return error


def adaline(data, labels, learning_rate):
	num_data_points, num_attributes = data.shape
	weight_matrix = np.random.rand(4)
	iterations = 501
	errors = []
	for it in range(iterations):
		for i in range(num_data_points):
			inp = data[i]
			actual = labels[i]
			observed = np.matmul(inp, weight_matrix)
			weight_update = learning_rate * (actual - observed) * inp
			weight_matrix += weight_update
			weight_matrix /= sum(weight_matrix)
		if it % 25 == 0:
			error = adaline_error(data, weight_matrix, labels)
			errors.append(error)
	xtickslabel = [i*25 for i in range(21)]
	plt.plot(np.arange(len(errors)), errors)
	plt.title("ADALINE Least Squares Error Over Time")
	plt.xlabel("Iterations")
	plt.xticks(np.arange(len(errors)), xtickslabel)
	plt.ylabel("Error")
	plt.show()
	

def find_bmu(t, nodes, m):
    bmu_idx = np.array([0, 0])
    min_dist = np.inf
    for x in range(nodes.shape[0]):
        for y in range(nodes.shape[1]):
            w = nodes[x, y, :].reshape(m, 1)
            sq_dist = np.sum((w - t) ** 2)
            if sq_dist < min_dist:
                min_dist = sq_dist
                index = np.array([x, y])
    bmu = nodes[index[0], index[1], :].reshape(m, 1)
    return (bmu, index)


def one_dim_feature_map(data, num_nodes):
	centered_data = StandardScaler().fit_transform(data)
	covariance_mat = np.cov(np.transpose(centered_data))
	eigenval, eigenvec = np.linalg.eig(covariance_mat)
	eigs = [(eigenval[i], eigenvec[:, i]) for i in range(4)]
	sorted_eigs = sorted(eigs, reverse=True)
	pcs = [sorted_eigs[i][1] for i in range(2)]
	conversion = np.array(pcs)
	projection = centered_data.dot(np.transpose(conversion))
	data = np.transpose(projection)

	network_dimensions = np.array([num_nodes,1])
	n_iters = 10000
	init_learning_rate = 0.01
	m = data.shape[0]
	n = data.shape[1]

	nodes = np.random.random((network_dimensions[0], network_dimensions[1], m))

	init_radius = max(network_dimensions[0], network_dimensions[1]) / 2
	time_constant = n_iters / np.log(init_radius)
	for i in range(n_iters):
		t = data[:, np.random.randint(0, n)].reshape(np.array([m, 1]))
		bmu, bmu_index = find_bmu(t, nodes, m)
		r = init_radius * np.exp(-i / time_constant)
		l = init_learning_rate * np.exp(-i / n_iters)
		for x in range(nodes.shape[0]):
		    for y in range(nodes.shape[1]):
		        w = nodes[x, y, :].reshape(m, 1)
		        w_dist = np.sum((np.array([x, y]) - bmu_index) ** 2)
		        if w_dist <= r**2:
		            neighborhood = np.exp(-w_dist / (2* (r**2)))
		            new_w = w + (l * neighborhood * (t - w))
		            new_nodes = nodes[x, y, :]
		            nodes[x, y, :] = new_w.reshape(1, 2)

	final_nodes = np.squeeze(nodes, axis=1)
	plt.scatter(data[0], data[1])
	plt.plot(final_nodes[:, 0], final_nodes[:, 1], 'ok')
	plt.plot(final_nodes[:, 0], final_nodes[:, 1], 'k')
	plt.title("SOFM with 25 Nodes")
	plt.show()


def two_dim_feature_map():
	# data = np.transpose(np.random.multivariate_normal([0.5, 0.5], [[1,0],[0,1]], size=5000))
	predata = np.random.uniform(size=(2, 5000))
	def inside_cirle(a, b):
		return math.sqrt((a-0.5)**2 + (b-0.5)**2)
	n_iters = 10000
	data0 = []
	data1 = []
	for i in range(5000):
		if inside_cirle(predata[0,i], predata[1,i]) <= 0.5:
			data0.append(predata[0,i])
			# print(predata[0,i])
			data1.append(predata[1,i])
	data = np.stack((np.array(data0), np.array(data1)),axis=0)

	network_dimensions = np.array([10, 10])
	
	init_learning_rate = 0.1
	m = data.shape[0]
	n = data.shape[1]

	nodes = np.random.random((network_dimensions[0], network_dimensions[1], m))

	init_radius = max(network_dimensions[0], network_dimensions[1]) / 2
	time_constant = n_iters / np.log(init_radius)
	for i in range(n_iters):
		t = data[:, np.random.randint(0, n)].reshape(np.array([m, 1]))
		bmu, bmu_index = find_bmu(t, nodes, m)
		r = init_radius * np.exp(-i / time_constant)
		l = init_learning_rate * np.exp(-i / n_iters)
		for x in range(nodes.shape[0]):
		    for y in range(nodes.shape[1]):
		        w = nodes[x, y, :].reshape(m, 1)
		        w_dist = np.sum((np.array([x, y]) - bmu_index) ** 2)
		        if w_dist <= r**2:
		            neighborhood = np.exp(-w_dist / (2* (r**2)))
		            new_w = w + (l * neighborhood * (t - w))
		            new_nodes = nodes[x, y, :]
		            nodes[x, y, :] = new_w.reshape(1, 2)
	
	nodes_2 = np.transpose(nodes)
	plt.plot(data[0], data[1], 'o', markersize=2)
	plt.plot(nodes[:, :, 0], nodes[:, :, 1], 'ok')
	plt.plot(nodes[:, :, 0], nodes[:, :, 1], 'k')
	plt.plot(nodes_2[0, :, :], nodes_2[1, :, :], 'k')
	plt.title(f"SOFM Grid, {n_iters} Iterations")
	plt.show()


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

	fig = plt.figure(num="Plots of Iris Dataset in varying dimensions")
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
	learning_rate = 0.001
	non_setosa_data = np.concatenate((versicolor, virginica), axis=0)

	'''
	Uncomment the functions on each of the following lines to run the corresponding 
	functions. learning_rate above may need to be adjusted for each function.
	'''

	# perceptron(x, y, learning_rate)
	# plot_perceptrons(x, y, [0.01, 0.0001, 0.00001])
	# visualize(x, y, setosa, versicolor, virginica)
	#### setosa is linearly separable from the other two ####
	# sanger(x, learning_rate)
	# reduce_dims(x, y, non_setosa_labels)
	# one_dim_feature_map(x, 25)
	# two_dim_feature_map()
	# adaline(non_setosa_data, non_setosa_labels, learning_rate)
	# sofm()
