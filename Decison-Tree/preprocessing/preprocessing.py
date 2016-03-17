'''

Simple preprocessing
Convert the data into desired format
i.e., it will generate metadata about data and convert data into csv format
all the attributes will be numeric and last attribute will be the class.
'''




import csv
import numpy
from scipy import stats

def normalize(gd):
	gd = stats.zscore(gd, axis=0)
	'''
	mini = gd.min(axis=0)
	maxi = gd.max(axis=0)
	diff = maxi - mini
	gd -= mini
	gd /= diff
	'''
	return gd

def preprocess_tab(input_file, output_dir, fname):

	with open(input_file,"r") as f:
		num_lines = sum(1 for line in f if line != "\n")
	with open(input_file,"r") as f:
		line = f.readline()
		print line
		features = len(line.split()) - 1
	classdict = {}
	points = numpy.zeros(shape=(num_lines, features+1))
	print num_lines, features
	with open(input_file,"r") as f:
		entry = -1
		for row in f:
			row = row.split()
			if row:
				entry += 1
				for i in range(0,features):
					points[entry][i] = float(row[i])
				if row[features] in classdict:
					points[entry][features] = classdict[row[features]]
				else:
					d_len = len(classdict)
					classdict[row[features]] = d_len
					points[entry][features] = d_len
	print points
	data = points[:,0:features]
	category = points[:,features:features+1]
	data = normalize(data)
	print numpy.hstack([data, category])
	numpy.savetxt(output_dir+"/"+fname+".data", numpy.hstack([data, category]), delimiter=",")
	with open(output_dir+"/"+fname+".info","w") as f:
		f.write("points "+str(num_lines)+"\n")
		f.write("features "+str(features)+"\n")
		f.write("categories "+str(len(classdict))+"\n")
	with open(output_dir+"/"+fname+".dict", 'wb') as f:
		writer = csv.writer(f)
		for key in classdict:
			value = classdict[key]
			print key,value
			writer.writerow([key, value])

def preprocess_yeast(input_file, output_dir, fname):

	with open(input_file,"r") as f:
		num_lines = sum(1 for line in f if line != "\n")
	with open(input_file,"r") as f:
		line = f.readline()
		print line
		features = len(line.split()) - 1
	classdict = {}
	attdict = {}
	points = numpy.zeros(shape=(num_lines, features+1))
	print num_lines, features
	with open(input_file,"r") as f:
		entry = -1
		for row in f:
			row = row.split()
			if row:
				entry += 1
				if row[0] in attdict:
					points[entry][0] = attdict[row[0]]
				else:
					d_len = len(attdict)
					attdict[row[0]] = d_len
					points[entry][0] = d_len
				for i in range(1,features):
					points[entry][i] = float(row[i])
				if row[features] in classdict:
					points[entry][features] = classdict[row[features]]
				else:
					d_len = len(classdict)
					classdict[row[features]] = d_len
					points[entry][features] = d_len
	print points
	data = points[:,0:features]
	category = points[:,features:features+1]
	data = normalize(data)
	print numpy.hstack([data, category])
	numpy.savetxt(output_dir+"/"+fname+".data", numpy.hstack([data, category]), delimiter=",")
	with open(output_dir+"/"+fname+".info","w") as f:
		f.write("points "+str(num_lines)+"\n")
		f.write("features "+str(features)+"\n")
		f.write("categories "+str(len(classdict))+"\n")
	with open(output_dir+"/"+fname+".dict", 'wb') as f:
		writer = csv.writer(f)
		for key in classdict:
			value = classdict[key]
			print key,value
			writer.writerow([key, value])

def preprocess(input_file, output_dir, fname):

	with open(input_file,"rU") as f:
		num_lines = sum(1 for line in f if line != "\n")
	with open(input_file,"rU") as f:
		line = f.readline()
		print line
		features = len(line.split(',')) - 1
	classdict = {}

	points = numpy.zeros(shape=(num_lines, features+1))
	print num_lines, features
	with open(input_file,"rU") as f:
		reader = csv.reader(f)
		entry = -1
		for row in reader:
			print row
			if row:
				entry += 1
				for i in range(features):
					points[entry][i] = float(row[i])
				if row[features] in classdict:
					points[entry][features] = classdict[row[features]]
				else:
					d_len = len(classdict)
					classdict[row[features]] = d_len
					points[entry][features] = d_len
	print points
	data = points[:,0:features]
	category = points[:,features:features+1]
	data = normalize(data)
	print numpy.hstack([data, category])
	numpy.savetxt(output_dir+"/"+fname+".data", numpy.hstack([data, category]), delimiter=",")
	with open(output_dir+"/"+fname+".info","w") as f:
		f.write("points "+str(num_lines)+"\n")
		f.write("features "+str(features)+"\n")
		f.write("categories "+str(len(classdict))+"\n")
	with open(output_dir+"/"+fname+".dict", 'wb') as f:
		writer = csv.writer(f)
		for key in classdict:
			value = classdict[key]
			print key,value
			writer.writerow([key, value])


def zero_index_preprocess(input_file, output_dir, fname):
	with open(input_file,"r") as f:
		num_lines = sum(1 for line in f if line != "\n")
	with open(input_file,"r") as f:
		line = f.readline()
		print line
		features = len(line.split(',')) - 1

	classdict = {}

	points = numpy.zeros(shape=(num_lines, features+1))
	print num_lines, features
	with open(input_file,"r") as f:
		reader = csv.reader(f)
		entry = -1
		for row in reader:
			if row:
				entry += 1
				for i in range(1,features+1):
					points[entry][i] = float(row[i])
				if row[0] in classdict:
					points[entry][0] = classdict[row[0]]
				else:
					d_len = len(classdict)
					classdict[row[0]] = d_len
					points[entry][0] = d_len
	print points
	data = points[:,1:features+1]
	category = points[:,0:1]
	data = normalize(data)
	print numpy.hstack([data, category])
	numpy.savetxt(output_dir+"/"+fname+".data", numpy.hstack([data, category]), delimiter=",")
	with open(output_dir+"/"+fname+".info","w") as f:
		f.write("points "+str(num_lines)+"\n")
		f.write("features "+str(features)+"\n")
		f.write("categories "+str(len(classdict))+"\n")
	with open(output_dir+"/"+fname+".dict", 'wb') as f:
		writer = csv.writer(f)
		for key in classdict:
			value = classdict[key]
			print key,value
			writer.writerow([key, value])

def id_atstart(input_file, output_directory, name):
	with open(input_file, "rb") as f:
		data = numpy.loadtxt(f,delimiter=",")
	rows, column = data.shape
	data = data[:,1:column]
	with open(input_file+".corrected2", "w") as f:
		numpy.savetxt(f, data, delimiter=",")
	preprocess(input_file+".corrected2", output_directory, name)

if __name__ == "__main__":
	#preprocess("../datasets/iris/iris.data.txt.", "../cleaned/iris", "iris")
	#preprocess("../datasets/transfusion/transfusion.data.txt.", "../cleaned/transfusion", "transfusion")
	#zero_index_preprocess("../datasets/wine/wine.data", "../cleaned/wine", "wine")
	#id_atstart("../datasets/glass/glass.data", "../cleaned/glass", "glass")
	#zero_index_preprocess("../datasets/leaf/leaf.data", "../cleaned/leaf", "leaf")
	#preprocess("../datasets/haberman/haberman.data", "../cleaned/haberman", "haberman")
	#preprocess("../datasets/banknote/banknote.data", "../cleaned/banknote", "banknote")
	#zero_index_preprocess("../datasets/wdbc/wdbc.data", "../cleaned/wdbc", "wdbc")
	# not done
	#preprocess_tab("../datasets/seeds/seeds.data", "../cleaned/seeds", "seeds")
	#preprocess_yeast("../datasets/yeast/yeast.data", "../cleaned/yeast", "yeast")
	#preprocess("../datasets/pima/pima.data", "../cleaned/pima", "pima")
	#preprocess("../datasets/kidney/kidney.data", "../cleaned/kidney", "kidney")
	#zero_index_preprocess("../datasets/image/image.data", "../cleaned/image", "image")
	zero_index_preprocess("../datasets/leaf/leaf.data", "../cleaned/leaf", "leaf")
