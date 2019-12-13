from csv_read import *
from sklearn.svm import SVC

if __name__ == '__main__':

	infile = 'train.csv'
	outfile= 'test.csv'
	subfile= 'submission.csv'

	inp = read(infile)#[:100]

	print('Input reading done ...')

	out = read(outfile)#[:100]

	print('Output Reading done ...')
	
	train_x, train_y =data_label_split(inp)

	print('Data extracted ...')
	
	clf=SVC()

	clf.fit(train_x, train_y)

	print('Training done ...')

	test_y = clf.predict(out)

	print('Prediction done ...')

	index=np.arange(len(test_y))+1
	header=np.array([0,0], dtype='int32')
	final_out=np.vstack((index,test_y))

	out_with_header = np.vstack((header,final_out.T))

	write(out_with_header,subfile)
