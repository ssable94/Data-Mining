import numpy
from random import shuffle
from decisionTree import decision_tree


'''
Evaluator class checks the performance of the decision tree algorithm. It will calculate the results and store it in
files, which can be used for further analysis and generating graphs.
'''


class Evaluator(object):

	def __init__(self, given_path, given_name, given_method, given_k, given_beam_size):

		# Setting parameter values
		self.path = given_path
		self.name = given_name
		self.method = given_method
		self.cross_k_val = given_k
		self.beam_size = given_beam_size

		# Reading required files and datasets
		self.all_data, self.points, self.features, self.categories, self.class_reverse_dict, self.class_dict = self.get_data()

	def test_point(self, tree, p):
		tmp = tree
		while tmp.leaf == 0 and tmp.left is not None and tmp.right is not None:
			if self.all_data[p][tmp.feature] <= tmp.value:
				tmp = tmp.left
			else:
				tmp = tmp.right
		return tmp.category

	def test_point_advanced(self, tree, p):
		tmp = tree
		while tmp.leaf == 0 and tmp.left is not None and tmp.right is not None:
			if self.all_data[p][tmp.feature] <= tmp.value:
				tmp = tmp.left
			else:
				tmp = tmp.right
		return tmp.category, tmp.confidence

	def get_intm_acc(self, tree, train):
		total = 0
		correct = 0
		for p in train:
			total += 1
			correct += 1 if self.all_data[p][self.features] == self.test_point(tree, p) else 0
		return float(correct)/float(total)

	def keep_best_m(self, trees, train):
		accuracy = [0]*len(trees)
		for i in range(len(trees)):
			accuracy[i] = self.get_intm_acc(trees[i], train)
		accuracy = zip(accuracy, range(len(trees)))
		accuracy.sort(key=lambda x: x[0], reverse=True)
		accuracy_values = [a for a,b in accuracy]
		accuracy = [b for a,b in accuracy]
		if len(accuracy) > self.beam_size:
			accuracy = accuracy[:self.beam_size]
			accuracy_values = accuracy_values[:self.beam_size]
		new_trees = []
		for i in accuracy:
			new_trees.append(trees[i])
		return new_trees, accuracy_values

	def evaluate(self):
		# split contains list of lists i.e, list of [train, test]
		split = self.split_data(self.points)

		open("output/accuracy_%s_%s_%s.csv" % (self.name, self.method, self.beam_size), 'w').close()

		for train, test in split:

			# Generate initial set of trees
			dtree = decision_tree(self.all_data, self.features, self.categories, self.method, len(train), self.beam_size)

			trees, keep_going = dtree.initialize_trees(train)
			trees, prev_acc = self.keep_best_m(trees, train)

			acc_sum = [sum(prev_acc)]

			# Iterate till we can not improve further
			while keep_going:
				tmp_trees = []
				keep_going = False
				for tree in trees:
					new_trees, going = dtree.extend(tree)
					tmp_trees += new_trees
					keep_going = keep_going or going
				trees, curr_acc = self.keep_best_m(tmp_trees, train)
				acc_sum.append(sum(curr_acc))
				if len(acc_sum) > 2:
					if acc_sum[-1] - acc_sum[-3] <= 0:
						keep_going = False
				i = 0
				while i < len(curr_acc) and curr_acc[i] == 1:
					i += 1
				if i == len(curr_acc):
					keep_going = False
			# trees has forest of trees

			# As we have final set of trees, checking accuracy and generating results.
			self.get_accuracy(trees, test)

	def get_data(self):
		points, features, categories = 0, 0, 0
		with open(self.path+"\\"+self.name+".info") as f:
			for line in f:
				line.rstrip()
				line = line.split()
				if line[0] == "points":
					points = int(line[1])
				elif line[0] == "features":
					features = int(line[1])
				elif line[0] == "categories":
					categories = int(line[1])

		Data = numpy.empty(shape=(points, features+1), dtype=float)

		entered = 0
		with open(self.path+"\\"+self.name+".data") as f:
			for line in f:
				sample = map(float, line.rstrip().split(','))
				if sample:
					for i in range(features+1):
						Data[entered][i] = sample[i]
					entered += 1

		class_reverse_dict = {}
		class_dict = {}
		with open(self.path+"\\"+self.name+".dict") as dict_file:
			for line in dict_file:
				line = line.rstrip().split(",")
				if line:
					class_reverse_dict[int(line[1])] = line[0]
					class_dict[line[0]] = int(line[1])

		return Data, points, features, categories, class_reverse_dict, class_dict

	def split_data(self, points):
		indices = range(points)
		shuffle(indices)
		n = -(-points/self.cross_k_val)

		split = []
		for i in xrange(0, points, n):
			test = indices[i:i+n]
			train = [p for p in indices if p not in test]
			split.append([train, test])
		return split

	def get_accuracy(self, trees, test):
		acc_set = numpy.empty(shape=(len(test), 2+self.categories), dtype=float)
		len_trees = float(len(trees))
		for p in range(len(test)):
			score = 0.0
			vector = [0]*self.categories
			for tree in trees:
				predicted, confidence = self.test_point_advanced(tree, test[p])
				for i in range(self.categories):
					vector[i] += confidence[i]
			acc_set[p][0] = vector.index(max(vector))											# category_given
			acc_set[p][1] = self.all_data[test[p]][self.features]								# actual category
			for i in range(self.categories):
				acc_set[p][2+i] = vector[i]														# confidence

		# Store the results in the file, so that file can be used in future.
		with open("output/accuracy_%s_%s_%s.csv" % (self.name, self.method, self.beam_size), "a") as f:
			numpy.savetxt(f, acc_set, delimiter=",")

	def get_roc_data(self, cat, acc_data):
		values = sorted(list(set(numpy.array(acc_data[:,2:3]).reshape(-1,).tolist())))

		# X-axis -> False Positive
		# Y-axis -> True Positive
		# 0 - TP
		# 1 - FP
		# 2 - TN
		# 3 - FN
		confusion_matrices = numpy.zeros(shape=(len(values),4), dtype=int)
		for i in range(len(values)):
			for row in acc_data:
				given_cat = -1
				if row[2] >= values[i]:
					given_cat = cat
				if row[1] == cat:
					if given_cat == cat:
						confusion_matrices[i][0] += 1
					else:
						confusion_matrices[i][3] += 1
				else:
					if given_cat != cat:
						confusion_matrices[i][2] += 1
					else:
						confusion_matrices[i][1] += 1
		roc_points = []
		for row in confusion_matrices:
			# sensitivity, specificity = float(row[0])/float(row[0]+row[3]), float(row[2])/float(row[2]+row[1])
			tp_rate, fp_rate = float(row[0])/float(row[0]+row[3]), float(row[1])/float(row[2]+row[1])
			roc_points.append((fp_rate, tp_rate))
		return roc_points

	def get_measures(self):

		acc_data = numpy.genfromtxt("output/accuracy_%s_%s_%s.csv" % (self.name, self.method, self.beam_size), delimiter=',')

		# calculating confusion matrix
		confusion_matrix = numpy.zeros(shape=(self.categories, 4), dtype=int)
		# 0 - TP
		# 1 - FP
		# 2 - TN
		# 3 - FN
		for cat in range(self.categories):
			for row in acc_data:
				if row[1] == cat:
					if row[0] == cat:
						confusion_matrix[cat][0] += 1
					else:
						confusion_matrix[cat][3] += 1
				else:
					if row[0] != cat:
						confusion_matrix[cat][2] += 1
					else:
						confusion_matrix[cat][1] += 1

		# calculating f1 score
		f1_score = []
		for cat in range(self.categories):
			if float(confusion_matrix[cat][0]+confusion_matrix[cat][1]) != 0:
				precision = float(confusion_matrix[cat][0])/float(confusion_matrix[cat][0]+confusion_matrix[cat][1])
			else:
				precision = 0
			if float(confusion_matrix[cat][0]+confusion_matrix[cat][3]) != 0:
				recall = float(confusion_matrix[cat][0])/float(confusion_matrix[cat][0]+confusion_matrix[cat][3])
			else:
				recall = 0
			if (precision+recall) != 0:
				f1 = 2*precision*recall/(precision+recall)
			else:
				f1 = 0
			f1_score.append(f1)
			f1_score.append(f1)
		# Storing f1 score
		with open("output/f1_score.csv", 'a') as f:
			for cat in range(self.categories):
				f.write(self.name+","+self.method+","+self.class_reverse_dict[cat]+","+str(f1_score[cat])+"\n")

		# Storing balanced accuracy
		with open("output/balanced_accuracy.csv", 'a') as f:
			for cat in range(self.categories):
				if (confusion_matrix[cat][0]+confusion_matrix[cat][3]) > 0:
					sensitivity = float(confusion_matrix[cat][0])/float(confusion_matrix[cat][0]+confusion_matrix[cat][3])
				else:
					sensitivity = 0
				if (confusion_matrix[cat][2]+confusion_matrix[cat][1]) > 0:
					specificity = float(confusion_matrix[cat][2])/float(confusion_matrix[cat][2]+confusion_matrix[cat][1])
				else:
					specificity = 0
				balanced_accuracy = float(sensitivity + specificity)/2.0
				f.write(self.name+","+self.method+","+self.class_reverse_dict[cat]+","+str(balanced_accuracy)+"\n")


		# Calculating simple accuracy using confusion matrix
		simple_accuracy = 0
		for cat in range(self.categories):
			simple_accuracy += confusion_matrix[cat][0]
		simple_accuracy = float(simple_accuracy)/float(acc_data.shape[0])
		# Storing accuracy
		with open("output/accuracy.csv","a") as f:
			f.write(self.name+","+self.method+","+str(simple_accuracy)+","+str(self.beam_size)+"\n")

		# Calculating data for ROC curve
		roc_data = []
		for cat in range(self.categories):
			roc_data.append(self.get_roc_data(cat, acc_data))
		# Storing roc_data
		with open("output/roc.csv",'a') as f:
			for cat in range(self.categories):
				for x,y in roc_data[cat]:
					f.write(self.name+","+self.method+","+str(cat)+","+str(x)+","+str(y)+"\n")


if __name__ == "__main__":

	# Erasing the contents of the file in which we want to store the new results.
	with open("output/roc.csv",'w') as f:
		f.write("name, method, class, x, y\n")
	with open("output/f1_score.csv", 'w') as f:
		f.write("name, method, class, f1_score\n")
	with open("output/balanced_accuracy.csv", 'w') as f:
		f.write("name, method, class, balanced_accuracy\n")

	# Read the input file parse it and run the algorithm
	with open("input.txt", "r") as f:
		for line in f:
			line = line.rstrip()
			line = line.split()
			if line:

				# line[0] - path of the directory
				# line[1] - name of the data set
				# line[2] - method
				# line[3] - k for k-cross validation
				# line[4] - beam size

				# Generating Evaluator instance for evaluating decision tree
				evaluator = Evaluator(line[0], line[1], line[2], int(line[3]), int(line[4]))

				# Performing evaluation
				evaluator.evaluate()

				# Calculating various performance measures
				evaluator.get_measures()

				# Showing status
				print "Completed", line[1], "Dataset using", "Information Gain" if line[2] == 'i' else "Gini"

