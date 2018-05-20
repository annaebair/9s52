import numpy as np
import random
import copy
import matplotlib.pyplot as plt

class Hopfield():

	def __init__(self, num_nodes, num_prototypes, num_test):
		self.num_nodes = num_nodes
		self.num_prototypes = num_prototypes
		self.vectors = self.generate_vectors(num_prototypes)
		self.test_vecs = self.generate_vectors(num_test)


	def generate_vectors(self, num_vectors):
		vectors = np.zeros((num_vectors, self.num_nodes))
		for i in range(num_vectors):
			random_vec = np.random.randint(2, size=self.num_nodes)
			random_vec = random_vec * 2 - 1
			vectors[i] = random_vec
		return vectors


	def hamming_distance(self, a, b):
		s = 0
		for i in range(len(a)):
			if a[i] != b[i]: s += 1
		return s


	def normalize(self, vec):
		for i in range(len(vec)):
			elt = vec[i]
			if elt < 0: vec[i] = -1
			else: vec[i] = 1
		return vec


	def get_first_n_prototypes(self, n):
		return self.vectors[:n]


	def set_weights(self, num_patterns):
		vecs = self.get_first_n_prototypes(num_patterns)
		weights = np.dot(vecs.T, vecs) / self.num_nodes
		return weights


	def test(self, num_patterns):
		test = copy.deepcopy(self.test_vecs)
		weights = self.set_weights(num_patterns)
		num_iters = 0
		while True:
			num_iters += 1
			stable_vecs = 0
			for vec in test:
				stable = 0
				nodes = np.arange(len(weights))
				np.random.shuffle(nodes)
				for node_idx in nodes:
					prod = np.dot(weights, vec)
					prod = self.normalize(prod)
					if prod[node_idx] == vec[node_idx]:
						stable += 1
					else:
						vec[node_idx] = prod[node_idx]
				if stable == len(weights):
					stable_vecs += 1
			if stable_vecs == len(test): 
				same, opp, lst = self.num_converged(test, num_patterns)
				return test, num_iters, same, opp, lst


	def num_converged(self, converged_vecs, num_patterns):
		vectors = self.get_first_n_prototypes(num_patterns)
		same = 0
		opp = 0
		closest_list = []
		for vec in converged_vecs:
			closest = 51
			for v in vectors:
				ham = self.hamming_distance(v, vec)
				if ham < closest: closest = ham
				if ham == 0:
					same += 1
				elif ham == 50:
					opp += 1
			closest_list.append(closest)
		return same, opp, closest_list


	def generate_hamming_graph(self):
		hamming_dists = np.zeros((self.num_prototypes, self.num_prototypes))
		for i in range(self.num_prototypes):
			for j in range(self.num_prototypes):
				hamming_dists[i][j] = self.hamming_distance(self.vectors[i], self.vectors[j])

		plt.imshow(hamming_dists, origin='lower')
		plt.title("Hamming distances between \n pairs of prototype vectors")
		plt.colorbar()
		plt.show()


	def generate_viz_graph(self):
		plt.imshow(self.vectors, origin='lower')
		plt.title("Visual representation of prototype vectors")
		plt.colorbar(shrink=0.5, ticks=[-1, 1], aspect=10)
		plt.xlabel("Nodes")
		plt.ylabel("Vectors")
		plt.xticks([0, 10, 20, 30, 40, 50])
		plt.show()


if __name__ == "__main__":
	hopfield = Hopfield(50, 14, 25)
	subplotrefs = [221, 222, 223, 224]
	patterns = [1, 4, 7, 14]
	for ind in range(4):
		i = patterns[ind]
		test, num_iters, same, opp, lst = hopfield.test(i)
		print(i, " patterns converged in ", num_iters, "iterations, same: ", same, ", opposite: ", opp)
		plt.subplot(subplotrefs[ind])
		plt.scatter(np.arange(25), lst)
		plt.ylabel("Distance")
		plt.ylim(-1, 51)
		plt.xlabel("Test Vector")
		if i == 1:
			plt.title(f"{i} Stored Pattern")
		else: 
			plt.title(f"{i} Stored Patterns")
	plt.subplots_adjust(hspace=0.5, wspace=0.3)
	# Uncomment lines below to show plots
	# plt.show()
	# hopfield.generate_hamming_graph()
	# hopfield.generate_viz_graph()
