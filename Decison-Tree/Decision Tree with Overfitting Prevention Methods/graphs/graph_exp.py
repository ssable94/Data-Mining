import numpy as np
import matplotlib.pyplot as plt


def plot(acc, majors):

	radius = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
	area = [3.14159, 12.56636, 28.27431, 50.26544, 78.53975, 113.09724]
	width = 0.25
	fig = plt.figure(figsize=(10,8))
	plt.plot(radius, area, color = 'red', label='bla')
	plt.plot(radius, [i/2 for i in area], color = np.random.rand(3,1))
	plt.plot(radius, [i/3 for i in area], color = 'green')
	#plt.bar([i+width for i in range(len(acc))], acc2, width, color="red", align= 'center', label = 'acc2')
	#plt.xticks(range(len(acc)),majors)
	plt.xlabel('Method Used')
	plt.ylabel('Accuracy')
	plt.title('Comparison Between Methods Based on Accuracy for Dataset %s' % "iris")
	'''plt.ylim([0,1])
	plt.xlim([0,1])
'''
	plt.legend()

	plt.show()
	fig.savefig('raw/fig_%s' % "iris", bbox_inches='tight')
	'''
	plt.bar(majors, acc)
	#plt.xticks( 0.35, majors)
	plt.show()

	n_groups = 5

	means_men = (20, 35, 30, 35, 27)
	std_men = (2, 3, 4, 1, 2)

	means_women = (25, 32, 34, 20, 25)
	std_women = (3, 5, 2, 3, 3)

	fig, ax = plt.subplots()

	index = np.arange(n_groups)
	bar_width = 0.35

	opacity = 0.4
	error_config = {'ecolor': '0.3'}

	rects1 = plt.bar(index, means_men, bar_width,
					 alpha=opacity,
					 color='b',
					 yerr=std_men,
					 error_kw=error_config,
					 label='Men')

	rects2 = plt.bar(index + bar_width, means_women, bar_width,
					 alpha=opacity,
					 color='r',
					 yerr=std_women,
					 error_kw=error_config,
					 label='Women')

	plt.xlabel('Group')
	plt.ylabel('Scores')
	plt.title('Scores by group and gender')
	plt.xticks(index + bar_width, ('A', 'B', 'C', 'D', 'E'))
	plt.legend()

	plt.tight_layout()
	plt.show()
	'''

if __name__ == "__main__":
	acc = [.89,.99,.65]
	majors = ["Pessimistic Estimation", "Validation Set", "DML"]
	plot(acc, majors)