import numpy
from random import shuffle
from decisionTree import decision_tree as generate_decision_tree
from decisionTree import tree_node
from exp import bytwo as beetwo


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
	if tmp.category == Data[p][features]:
		return True
	else:
		return False


def get_accuracy(Decision_tree, Data, test, features):
	total = 0
	correct = 0
	for p in test:
		total += 1
		if test_point(Decision_tree,Data, p, features):
			correct += 1
	print "tick"
	return float(correct)/float(total)


def evaluate(path, name, method, k):
	Data, points_index, points, features, categories = get_data(path, name, method)
	split = split_data(points, k)
	# split contains list of lists i.e, list of [train, test]

	accuracy = []

	for train, test in split:
		decision_tree = generate_decision_tree(Data, features, train, method, categories)
		accuracy.append(get_accuracy(decision_tree,Data, test, features))

	return float(sum(accuracy))/float(len(accuracy))


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

				with open("output.txt","a") as f:
					print "starting"
					accuracy = evaluate(line[0], line[1], line[2], int(line[3]))
					f.write(line[1]+","+line[2]+","+str(accuracy)+"\n")
					print "done with", line[1]


















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