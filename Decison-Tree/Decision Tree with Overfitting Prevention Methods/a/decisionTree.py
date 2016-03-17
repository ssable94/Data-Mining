import numpy
import math


class decision_tree(object):

	class tree_node(object):
		def __init__(self):
			self.leaf = 0
			self.category = -1
			self.confidence = []

			self.feature = 0
			self.value = -1
			self.right = None
			self.left = None

	def __init__(self, given_Data, given_features, given_categories, given_method, len_train):
		self.data = given_Data
		self.features = given_features
		self.categories = given_categories
		self.method = given_method
		self.tc = len_train

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

	def best_attr(self,points_index):
		attr_data = []
		for i in range(self.features):
			major, val = self.get_mandv(i, points_index)
			attr_data.append([i, major, val])
		attr_data.sort(key=lambda x: x[1], reverse=False)
		if attr_data[0][1] == 1000:
			return -1, -1
		return attr_data[0][0], attr_data[0][2]

	def get_majorClass_and_count(self, points_index):
		cat_list = [0]*self.categories
		for i in points_index:
			cat_list[int(self.data[i][self.features])] += 1
		maxi = max(cat_list)
		return cat_list.index(maxi), maxi, cat_list

	def build_decision_tree(self, points_index):

		l_points_index = float(len(points_index))

		# Generated new node
		node = self.tree_node()
		node.leaf = 1
		node.category, class_majority_count, node.confidence = self.get_majorClass_and_count(points_index)
		node.confidence = [float(i)/l_points_index for i in node.confidence]

		if node.confidence == 1:
			return node

		attr, value = self.best_attr(points_index)

		if attr == -1:
			return node

		right_pt, left_pt = [], []
		for i in points_index:
			if self.data[i][attr] <= value:
				left_pt.append(i)
			else:
				right_pt.append(i)

		p_mc = self.get_majorClass_and_count(points_index)[1]
		l_mc = self.get_majorClass_and_count(left_pt)[1]
		r_mc = self.get_majorClass_and_count(right_pt)[1]

		if r_mc + l_mc > p_mc:
			flag = True
		else:
			flag = False

		if flag:
			node.leaf = 0
			node.feature = attr
			node.value = value
			node.left = self.build_decision_tree(left_pt)
			node.right = self.build_decision_tree(right_pt)
		else:
			# print "Pruned by DML approach"
			return node
		return node

# end of code
































'''
	def build_decision_tree(self, points_index):

		# Generated new node
		node = self.tree_node()

		# checking for termination condition
		distinct_categories = set()
		for i in points_index:
			distinct_categories.add(self.data[i][self.features])
		distinct_categories = list(distinct_categories)

		if len(distinct_categories) == 1:
			node.leaf = 1
			node.category = distinct_categories[0]
			return node

		major_class = self.get_majorClass(points_index)

		attr, value = self.best_attr(points_index)
		if attr == -1:
			node.leaf = 1
			node.category = major_class
			return node
		node.leaf = 0
		node.feature = attr
		node.value = value
		right_pt, left_pt = [], []
		for i in points_index:
			if self.data[i][attr] <= value:
				left_pt.append(i)
			else:
				right_pt.append(i)

		l_len = len(left_pt)
		r_len = len(right_pt)

		if l_len != 0 and r_len != 0:

			flag = self.get_dml_verdict(points_index, left_pt, right_pt)

			if flag:
				node.left = self.build_decision_tree(left_pt)
				node.right = self.build_decision_tree(right_pt)
			else:
				print "Pruned by DML approach"
				# print correct_right, correct_left, correct_parent
				node.leaf = 1
				node.category = major_class
				return node
		elif l_len == 0:
			node = self.build_decision_tree(right_pt)
		elif r_len == 0:
			node = self.build_decision_tree(left_pt)
		return node
'''