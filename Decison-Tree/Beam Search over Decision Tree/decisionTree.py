import numpy
import math


class decision_tree(object):

	class tree_node(object):
		def __init__(self):
			self.leaf = 0
			self.category = -1
			self.confidence = []
			self.points_index = []
			self.left_points_index = []
			self.right_points_index = []
			self.correct_count = 0

			self.feature = 0
			self.value = -1
			self.right = None
			self.left = None

	def __init__(self, given_Data, given_features, given_categories, given_method, len_train, given_beam_size):
		self.data = given_Data
		self.features = given_features
		self.categories = given_categories
		self.method = given_method
		self.beam_size = given_beam_size

		self.node_cost = math.log(given_features)
		self.leaf_cost = math.log(given_categories)
		self.error_cost = math.log(len_train)

	def get_entropy(self, catlist):

		l_cat = len(catlist)

		if l_cat == 0:
			return 0
		catlist = map(int, catlist)
		major = 0.0
		cat = [0] * self.categories

		for i in catlist:
			cat[i] += 1

		for i in cat:
			k = float(i)/float(l_cat)
			if k != 0:
				major -= k*math.log(k)
		return major

	def get_gini(self,catlist):

		l_cat = len(catlist)

		if l_cat == 0:
			return 0
		catlist = map(int, catlist)
		major = 0.0
		cat = [0] * self.categories

		for i in catlist:
			cat[i] += 1

		for i in cat:
			major += i*i
		major /= float(l_cat*l_cat)
		return 1.0-major

	def get_values(self,attr, points_index):

		vals = [self.data[i][attr] for i in points_index]
		vals.sort()
		parts = int(math.ceil(math.log(len(vals),2)))
		#parts = len(vals)/2
		#parts = len(vals)/2
		#vals = list(set(vals[::parts]))
		vals = list(set(vals))
		vals.sort()
		return vals

	def get_mandv(self, attr, points_index):
		values = self.get_values(attr, points_index)

		mandv = []

		for value in values:
			l_set = [self.data[pt][self.features] for pt in points_index if self.data[pt][attr] <= value]
			r_set = [self.data[pt][self.features] for pt in points_index if self.data[pt][attr] > value]

			left_len = len(l_set)
			right_len = len(r_set)
			major = 0
			if left_len == 0 or right_len == 0:
				major = 1000
			else:
				if self.method == "g":
					major = self.get_gini(r_set)*len(r_set) + self.get_gini(l_set)*len(l_set)
				if self.method == "i":
					major = self.get_entropy(r_set)*len(r_set) + self.get_gini(l_set)*len(l_set)
				major = float(major)/float(len(points_index))
			mandv.append([major, value])
		mandv.sort(key=lambda x:x[0])
		return mandv[0][0], mandv[0][1]

	def attr_list(self,points_index):
		attr_data = []
		for i in range(self.features):
			major, val = self.get_mandv(i, points_index)
			if major != 1000:
				attr_data.append([i, major, val])
		attr_data.sort(key=lambda x: x[1], reverse=False)
		return attr_data

	def get_majorClass_and_count(self, points_index):
		cat_list = [0]*self.categories
		for i in points_index:
			cat_list[int(self.data[i][self.features])] += 1
		maxi = max(cat_list)
		return cat_list.index(maxi), maxi, cat_list

	def get_dml_verdict(self, points_index, left_pt, right_pt):
		pc = self.get_majorClass_and_count(points_index)[1]
		lc = self.get_majorClass_and_count(left_pt)[1]
		rc = self.get_majorClass_and_count(right_pt)[1]

		decrease = (float(rc + lc - pc))*self.error_cost - self.node_cost
		if decrease >= 0:
			return True
		else:
			return False

	def is_duplicate(self, node1, node2):

		if node1.leaf != node2.leaf:
			return 0

		if node1.leaf == 1:
			if node1.feature == node2.feature:
				return 1
			else:
				return 0

		if node1.feature == node2.feature and node1.value == node2.value:
			if node1.left is None and node2.left is None and node1.right is None and node2.right is None:
				return 1
			elif node1.left is None or node2.left is None or node1.right is None or node2.right is None:
				return 0
			if (self.is_duplicate(node1.left, node2.left)+self.is_duplicate(node1.left, node2.left)) == 2:
				return 1
			else:
				return 0
		else:
			return 0

	def remove_duplicate_trees(self,trees):
		unique = []
		for i in range(len(trees)):
			j = i+1
			foundmatch = 0
			while j < len(trees) and foundmatch == 0:
				foundmatch = self.is_duplicate(trees[i], trees[j])
				j += 1
			if foundmatch == 0:
				unique.append(trees[i])
		return unique

	def get_duplicate(self, node):
		new_node = self.tree_node()
		new_node.leaf = node.leaf
		new_node.category = node.category
		new_node.confidence = node.confidence[:]
		new_node.points_index = node.points_index[:]
		new_node.left_points_index = node.left_points_index[:]
		new_node.right_points_index = node.right_points_index[:]
		new_node.feature = node.feature
		new_node.value = node.value
		new_node.right = node.right
		new_node.left = node.left
		return new_node

	def extend(self, node):
		if node.leaf == 1:
			return [node], False

		if node.left is None:
			left_nodes, left_going = self.initialize_trees(node.left_points_index)
		else:
			left_nodes, left_going = self.extend(node.left)\

		if node.right is None:
			right_nodes, right_going = self.initialize_trees(node.right_points_index)
		else:
			right_nodes, right_going = self.extend(node.right)

		new_trees = []
		correct_count = []
		for right_node in right_nodes:
			for left_node in left_nodes:
				n_node = self.get_duplicate(node)
				n_node.right = right_node
				n_node.left = left_node
				n_node.correct_count = right_node.correct_count + left_node.correct_count
				correct_count.append(n_node.correct_count)
				new_trees.append(n_node)

		# below is the code to prun the tree, I am sure that it works perfectly fine and do not alter how beam search
		# works, but due to lack of time, I couldn't prove it mathemaically.

		# thus, if you want you can simply comment the code below, and uncommnet last line about return statement,
		# it will not prun the trees and it will work perfectly as beam search, though I am sure my apporach works
		# i do not have enough time for mathematical prove

		correct_count = zip(correct_count, range(len(correct_count)))
		correct_count.sort(key=lambda x:x[0], reverse=True)
		correct_count_indices = [b for a,b in correct_count]
		if len(correct_count_indices) > self.beam_size:
			correct_count_indices = correct_count_indices[:self.beam_size]
		top_trees = []
		for i in correct_count_indices:
			top_trees.append(new_trees[i])

		#top_trees = new_trees
		return top_trees, left_going or right_going

	def initialize_trees(self, points_index):

		l_points_index = float(len(points_index))

		# Generated new node
		node = self.tree_node()
		node.point_index = points_index[:]
		node.leaf = 1
		node.category, class_majority_count, node.confidence = self.get_majorClass_and_count(points_index)
		node.correct_count = class_majority_count
		node.confidence = [float(i)/l_points_index for i in node.confidence]
		if node.confidence == 1:
			return [node], False

		attr_list = self.attr_list(points_index)

		if len(attr_list) == 0:
			return [node], False

		if len(attr_list) > self.beam_size:
			attr_list = attr_list[:self.beam_size]

		nodes = []
		category = node.category
		confidence = node.confidence

		for attr, major, value in attr_list:

			right_pt, left_pt = [], []
			for i in points_index:
				if self.data[i][attr] <= value:
					left_pt.append(i)
				else:
					right_pt.append(i)

			#if self.get_dml_verdict(points_index, left_pt, right_pt):
			node = self.tree_node()
			node.points_index = points_index[:]
			node.leaf = 0
			node.category = category
			node.confidence = confidence
			node.feature = attr
			node.value = value
			node.left = None
			node.right = None
			node.correct_count = class_majority_count
			node.left_points_index = left_pt[:]
			node.right_points_index = right_pt[:]
			nodes.append(node)
		if nodes:
			return nodes, True
		else:
			return [node], False
