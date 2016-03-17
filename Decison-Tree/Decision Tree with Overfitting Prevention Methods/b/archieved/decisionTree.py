import numpy
import math

class tree_node(object):

	def __init__(self):
		self.leaf = 0
		self.category = -1
		self.confidence = -1

		self.feature = 0
		self.value = -1
		self.right = None
		self.left = None


def get_entropy(catlist, categories):

	l_cat = len(catlist)

	if l_cat == 0:
		return 0
	catlist = map(int, catlist)
	major = 0.0
	cat = [0] * categories

	for i in catlist:
		cat[i] += 1

	for i in cat:
		k = float(i)/float(l_cat)
		if k != 0:
			major -= k*math.log(k)
	return major


def get_gini(catlist, categories):

	l_cat = len(catlist)

	if l_cat == 0:
		return 0
	catlist = map(int, catlist)
	major = 0.0
	cat = [0] * categories

	for i in catlist:
		cat[i] += 1

	for i in cat:
		major += i*i
	major /= float(l_cat*l_cat)
	return 1.0-major

def get_values(data, attr, points_index):

	vals = [data[i][attr] for i in points_index]
	vals.sort()
	parts = int(math.ceil(math.log(len(vals),2)))
	#parts = len(vals)/2
	#parts = len(vals)/2
	#vals = list(set(vals[::parts]))
	vals = list(set(vals))
	vals.sort()
	return vals


def get_mandv(data, features, attr, points_index, method, categories):
	values = get_values(data, attr, points_index)

	mandv = []

	for value in values:
		l_set = [data[pt][features] for pt in points_index if data[pt][attr] <= value]
		r_set = [data[pt][features] for pt in points_index if data[pt][attr] > value]

		left_len = len(l_set)
		right_len = len(r_set)
		major = 0
		if left_len == 0 or right_len == 0:
			major = 1000
		else:
			if method == "g":
				major = get_gini(r_set, categories)*len(r_set) + get_gini(l_set, categories)*len(l_set)
			if method == "i":
				major = get_entropy(r_set, categories)*len(r_set) + get_gini(l_set, categories)*len(l_set)
			major = float(major)/float(len(points_index))
		mandv.append([major, value])
	mandv.sort(key=lambda x:x[0])
	return mandv[0][0], mandv[0][1]


def best_attr(data, features, points_index, method, categories):
	attr_data = []
	for i in range(features):
		major, val = get_mandv(data, features, i, points_index, method, categories)
		attr_data.append([i, major, val])
	attr_data.sort(key=lambda x: x[1], reverse=False)
	if attr_data[0][1] == 1000:
		return -1, -1
	return attr_data[0][0], attr_data[0][2]


def get_majorClass(data, features, points_index, categories):
	cat_list = [0]*categories
	for i in points_index:
		cat_list[int(data[i][features])] += 1
	cat_list = zip(cat_list, range(categories))
	cat_list.sort(key=lambda x: x[0], reverse=True)
	return cat_list[0][1]


def decision_tree(data, features, points_index, method, categories, validation):

	# Generated new node to return
	node = tree_node()

	# checking for termination condition
	distinct_categories = set()
	for i in points_index:
		distinct_categories.add(data[i][features])
	distinct_categories = list(distinct_categories)

	# terminating condition found
	if len(distinct_categories) == 1:
		node.leaf = 1
		node.category = distinct_categories[0]
		return node

	major_class = get_majorClass(data, features, points_index, categories)

	attr, value = best_attr(data, features, points_index, method, categories)
	if attr == -1:
		node.leaf = 1
		node.category = major_class
		return node
	node.leaf = 0
	node.feature = attr
	node.value = value
	right_pt, left_pt = [], []
	for i in points_index:
		if data[i][attr] <= value:
			left_pt.append(i)
		else:
			right_pt.append(i)

	left_validation = []
	right_validation = []

	for i in validation:
		if data[i][attr] <= value:
			left_validation.append(i)
		else:
			right_validation.append(i)

	l_len = len(left_pt)
	r_len = len(right_pt)

	if l_len != 0 and r_len != 0:
		correct_parent = 0
		for i in validation:
			if data[i][features] == major_class:
				correct_parent += 1

		left_major_class = get_majorClass(data, features, left_pt, categories)
		correct_left = 0
		for i in left_validation:
			if data[i][features] == left_major_class:
				correct_left += 1

		right_major_class = get_majorClass(data, features, right_pt, categories)
		correct_right = 0
		for i in right_validation:
			if data[i][features] == right_major_class:
				correct_right += 1

		#print "*"
		#print correct_parent, correct_left, correct_right, correct_left + correct_right
		#print major_class, left_major_class, right_major_class
		#print len(points_index), len(right_pt), len(left_pt)

		if (correct_right + correct_left) > correct_parent:
			node.left = decision_tree(data, features, left_pt, method, categories, left_validation)
			node.right = decision_tree(data, features, right_pt, method, categories, right_validation)
		else:
			# print "Pruned by validation approach"
			# print correct_right, correct_left, correct_parent
			node.leaf = 1
			node.category = major_class
			return node
	elif l_len == 0:
		node = decision_tree(data, features, right_pt, method, categories, validation)
	elif r_len == 0:
		node = decision_tree(data, features, left_pt, method, categories, validation)

	return node


























'''
def best_attr(data, features, points_index, method, categories):

	# random approach for testing purpose
	random_attr = randint(0, features-1)
	count = 0
	sumi = 0
	for i in points_index:
		count += 1
		sumi += data[i][random_attr]
	if count != 0:
		val = sumi/count
	else:
		val = sumi
	return random_attr, val
'''