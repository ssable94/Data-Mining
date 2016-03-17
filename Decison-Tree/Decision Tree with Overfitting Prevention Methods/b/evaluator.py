import numpy
from random import shuffle
from decisionTree import decision_tree
from random import randint

class evaluator(object):

	def __init__(self, given_path, given_name, given_method, given_k):
		# line[0] - path of the directory
		# line[1] - name of the data set
		# line[2] - method
		# line[3] - k for k-cross validation

		self.path = given_path
		self.name = given_name
		self.method = given_method
		self.cross_k_val = given_k

		self.all_data, self.points, self.features, self.categories, self.class_reverse_dict, self.class_dict = self.get_data()

	def evaluate(self):
		# split contains list of lists i.e, list of [train, test]
		split = self.split_data(self.points)

		open("output/accuracy_%s_%s.csv" % (self.name, self.method), 'w').close()

		for train, test in split:
			shuffle(train)
			k = int(float(len(train))*0.75)
			validation = train[k:]
			train = train[:k]
			dtree = decision_tree(self.all_data, self.features, self.categories, self.method, len(train))
			Decision_tree = dtree.build_decision_tree(train, validation)
			self.get_accuracy(Decision_tree, test)

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
			split.append([train,test])
		return split

	def test_point(self, Decision_tree, p):
		tmp = Decision_tree
		divide = 1.0
		while tmp.leaf == 0:
			divide += 0.2
			if self.all_data[p][tmp.feature] <= tmp.value:
				tmp = tmp.left
			else:
				tmp = tmp.right
		k = 1.0 - float(randint(0,10))/100
		confi = [(i/divide)*k for i in tmp.confidence]
		return tmp.category, self.all_data[p][self.features], confi #tmp.confidence[int(self.all_data[p][self.features])]/divide

	def get_accuracy(self, Decision_tree, test):
		acc_set = numpy.empty(shape=(len(test), 2+self.categories), dtype =float)
		total = 0
		for p in test:
			acc = self.test_point(Decision_tree, p)
			acc_set[total][0] = acc[0] # category given by decision tree
			acc_set[total][1] = acc[1] # actual category
			for i in range(self.categories):
				acc_set[total][2+i] = acc[2][i] # confidence
			total += 1

		with open("output/accuracy_%s_%s.csv" % (self.name, self.method), "a") as f:
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

		acc_data = numpy.genfromtxt("output/accuracy_%s_%s.csv" % (self.name, self.method), delimiter=',')

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
		# Storing f1 score
		with open("output/f1_score.csv", 'a') as f:
			for cat in range(self.categories):
				f.write(self.name+","+self.method+","+self.class_reverse_dict[cat]+","+str(f1_score[cat])+"\n")

		# Storing balanced accuracy
		with open("output/balanced_accuracy.csv", 'a') as f:
			for cat in range(self.categories):
				sensitivity = float(confusion_matrix[cat][0])/float(confusion_matrix[cat][0]+confusion_matrix[cat][3])
				specificity = float(confusion_matrix[cat][2])/float(confusion_matrix[cat][2]+confusion_matrix[cat][1])
				balanced_accuracy = float(sensitivity + specificity)/2.0
				f.write(self.name+","+self.method+","+self.class_reverse_dict[cat]+","+str(balanced_accuracy)+"\n")


		# Calculating simple accuracy using confusion matrix
		simple_accuracy = 0
		for cat in range(self.categories):
			simple_accuracy += confusion_matrix[cat][0]
		simple_accuracy = float(simple_accuracy)/float(acc_data.shape[0])
		# Storing accuracy
		with open("output/accuracy.csv","a") as f:
			f.write(self.name+","+self.method+","+str(simple_accuracy)+"\n")

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

	with open("output/roc.csv",'w') as f:
		f.write("name, method, class, x, y\n")
	with open("output/f1_score.csv", 'w') as f:
		f.write("name, method, class, f1_score\n")
	with open("output/balanced_accuracy.csv", 'w') as f:
		f.write("name, method, class, balanced_accuracy\n")

	with open("../input.txt", "r") as f:
		for line in f:
			line = line.rstrip()
			line = line.split()
			if line:

				# line[0] - path of the directory
				# line[1] - name of the data set
				# line[2] - method
				# line[3] - k for k-cross validation

				eval = evaluator(line[0], line[1], line[2], int(line[3]))
				eval.evaluate()
				eval.get_measures()
				print "done with ", line[1]

# end of code
