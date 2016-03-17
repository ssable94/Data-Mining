

import numpy
from random import shuffle
from decisionTree import decision_tree


def get_data(path, name, method):

	points, features, categories = 0, 0, 0
	with open(path+"\\"+name+".info") as f:
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
	with open(path+"\\"+name+".data") as f:
		for line in f:
			sample = map(float, line.rstrip().split(','))
			if sample:
				for i in range(features+1):
					Data[entered][i] = sample[i]
				entered += 1

	return Data, range(points), points, features, categories


def split_data(points, k):
	indices = range(points)
	shuffle(indices)
	n = -(-points/k)

	split = []
	for i in xrange(0, points, n):
		test = indices[i:i+n]
		train = [p for p in indices if p not in test]
		split.append([train,test])
	return split


def test_point(Decision_tree, Data, p, features):
	tmp = Decision_tree
	while tmp.leaf == 0:
		if Data[p][tmp.feature] <= tmp.value:
			tmp = tmp.left
		else:
			tmp = tmp.right
	return tmp.category, Data[p][features], tmp.confidence[int(Data[p][features])]


def get_accuracy(Decision_tree, Data, test, features, name, method):
	acc_set = numpy.empty(shape=(len(test), 3), dtype =float)
	total = 0
	for p in test:
		acc = test_point(Decision_tree,Data, p, features)
		acc_set[total][0] = acc[0] # category given by decision tree
		acc_set[total][1] = acc[1] # actual category
		acc_set[total][2] = acc[2] # confidence
		total += 1

	with open("output/accuracy_%s_%s.csv" % (name, method), "a") as f:
		print acc_set.shape
		numpy.savetxt(f, acc_set, delimiter=",")


def evaluate(path, name, method, k):
	Data, points_index, points, features, categories = get_data(path, name, method)
	split = split_data(points, k)

	open("output/accuracy_%s_%s.csv" % (name, method), 'w').close()
	# split contains list of lists i.e, list of [train, test]
	accuracy = []

	for train, test in split:
		dtree = decision_tree(Data, features, categories, method, len(train))
		Decision_tree = dtree.build_decision_tree(train)
		get_accuracy(Decision_tree,Data, test, features, name, method)


def get_measures(path, name, method):

	categories = 0
	with open(path+"\\"+name+".info") as f:
		for line in f:
			line.rstrip()
			line = line.split()
			if line[0] == "categories":
				categories = int(line[1])

	acc_data = numpy.genfromtxt("output/accuracy_%s_%s.csv" % (name, method), delimiter=',')

	confusion_matrix = numpy.zeros(shape=(categories, 4), dtype=int)
	# 0 - TP
	# 1 - FP
	# 2 - TN
	# 3 - FN
	for cat in range(categories):
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
	f1_score = []
	for cat in range(categories):
		precision = float(confusion_matrix[cat][0])/float(confusion_matrix[cat][0]+confusion_matrix[cat][1])
		recall = float(confusion_matrix[cat][0])/float(confusion_matrix[cat][0]+confusion_matrix[cat][3])
		f1 = 2*precision*recall/(precision+recall)
		f1_score.append(f1)
	print f1_score
	with open("output/f1socre_%s_%s.csv" % (name, method), 'w') as f:
		with open(path+"\\"+name+".dict") as dict_file:
			for line in dict_file:
				line = line.rstrip().split(",")
				print line
				f.write(line[0]+","+str(f1_score[int(line[1])])+"\n")

	print confusion_matrix
	simple_accuracy = 0
	for cat in range(categories):
		simple_accuracy += confusion_matrix[cat][0]
	simple_accuracy = float(simple_accuracy)/float(acc_data.shape[0])

	with open("output/accuracy.csv","a") as f:
		f.write(name+","+method+","+str(simple_accuracy)+"\n")


if __name__ == "__main__":
	with open("input.txt", "r") as f:
		for line in f:
			line = line.rstrip()
			line = line.split()
			if line:

				# line[0] - path of the directory
				# line[1] - name of the data set
				# line[2] - method
				# line[3] - k for k-cross validation

				evaluate(line[0], line[1], line[2], int(line[3]))
				get_measures(line[0], line[1], line[2])



# end of code

"""
import numpy
from random import shuffle
from decisionTree import decision_tree


def get_data(path, name, method):

	points, features, categories = 0, 0, 0
	with open(path+"\\"+name+".info") as f:
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
	with open(path+"\\"+name+".data") as f:
		for line in f:
			sample = map(float, line.rstrip().split(','))
			if sample:
				for i in range(features+1):
					Data[entered][i] = sample[i]
				entered += 1

	return Data, range(points), points, features, categories


def split_data(points, k):
	indices = range(points)
	shuffle(indices)
	n = -(-points/k)

	split = []
	for i in xrange(0, points, n):
		test = indices[i:i+n]
		train = [p for p in indices if p not in test]
		split.append([train,test])
	return split


def test_point(Decision_tree, Data, p, features):
	tmp = Decision_tree
	while tmp.leaf == 0:
		if Data[p][tmp.feature] <= tmp.value:
			tmp = tmp.left
		else:
			tmp = tmp.right
	return tmp.category, Data[p][features], tmp.confidence[int(Data[p][features])]


def get_accuracy(Decision_tree, Data, test, features, name, method):
	acc_set = numpy.empty(shape=(len(test), 3), dtype =float)
	total = 0
	correct = 0
	for p in test:
		acc = test_point(Decision_tree,Data, p, features)
		acc_set[total][0] = acc[0] # category given by decision tree
		acc_set[total][1] = acc[1] # actual category
		acc_set[total][2] = acc[2] # confidence

		total += 1
		if test_point(Decision_tree,Data, p, features):
			correct += 1

	with open("output/accuracy_%s_%s.csv" % (name, method), "a") as f:
		print acc_set.shape
		numpy.savetxt(f, acc_set, delimiter=",")
	return float(correct)/float(total)


def evaluate(path, name, method, k):
	Data, points_index, points, features, categories = get_data(path, name, method)
	split = split_data(points, k)

	open("output/accuracy_%s_%s.csv" % (name, method), 'w').close()
	# split contains list of lists i.e, list of [train, test]
	accuracy = []

	for train, test in split:
		dtree = decision_tree(Data, features, categories, method, len(train))
		Decision_tree = dtree.build_decision_tree(train)
		accuracy.append(get_accuracy(Decision_tree,Data, test, features, name, method))

	acc_data = numpy.genfromtxt("output/accuracy_%s_%s.csv" % (name, method), delimiter=',')
	confusion_matrix = numpy.zeros(shape=(categories, 4), dtype=int)
	# 0 - TP
	# 1 - FP
	# 2 - TN
	# 3 - FN
	for cat in range(categories):
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
	f1_score = []
	for cat in range(categories):
		precision = float(confusion_matrix[cat][0])/float(confusion_matrix[cat][0]+confusion_matrix[cat][1])
		recall = float(confusion_matrix[cat][0])/float(confusion_matrix[cat][0]+confusion_matrix[cat][3])
		f1 = 2*precision*recall/(precision+recall)
		f1_score.append(f1)
	print f1_score
	with open("output/f1socre_%s_%s.csv" % (name, method), 'w') as f:
		with open(path+"\\"+name+".dict") as dict_file:
			for line in dict_file:
				line = line.rstrip().split(",")
				print line
				f.write(line[0]+","+str(f1_score[int(line[1])])+"\n")

	return float(sum(accuracy))/float(len(accuracy))


def get_measures():


if __name__ == "__main__":
	with open("input.txt", "r") as f:
		for line in f:
			line = line.rstrip()
			line = line.split()
			if line:

				# line[0] - path of the directory
				# line[1] - name of the data set
				# line[2] - method
				# line[3] - k for k-cross validation

				with open("output/accuracy.csv","a") as f:
					accuracy = evaluate(line[0], line[1], line[2], int(line[3]))
					get_measures(line[0], line[1], line[2])
					f.write(line[1]+","+line[2]+","+str(accuracy)+"\n")

"""













"""
testing code for split works

from random import shuffle
def split_data(points, k):
	indices = range(points)
	shuffle(indices)
	n = -(-points/k)

	split = []
	for i in xrange(0, points, n):
		test = indices[i:i+n]
		train = [p for p in indices if p not in test]
		split.append([train,test])
	return split
split = split_data(150,10)
train = [0]*150
test = [0]*150
for a,b in split:
	for i in a:
		train[i] += 1
	for i in b:
		test[i] += 1
	print "train", a, len(a), "test", b, len(b)
print train, test

"""