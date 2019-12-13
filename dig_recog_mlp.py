from csv_read import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

if __name__=='__main__':

	#filenames
	infile = 'train.csv'
	outfile= 'test.csv'
	subfile= 'submission_mlp.csv'

	#MLP hyper-parameters
	hidden_units1 = 64
	hidden_units2 = 32
	lr = 0.001
	epochs = 50
	batch_size = 16

	inp = read(infile)

	print('Input reading done ...')

	out = read(outfile)

	print('Output Reading done ...')
	
	train_x, train_y =data_label_split(inp)
	train_y_onehot = keras.utils.to_categorical(train_y, num_classes=10)

	print('Data extracted ...')

	model = Sequential()

	model.add(Dense(hidden_units1, activation='relu', input_dim=784))
	model.add(Dense(hidden_units2, activation='relu'))
	model.add(Dense(10, activation='softmax'))

	sgd=SGD(lr=lr)

	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

	model.fit(train_x, train_y_onehot, epochs=epochs, batch_size=batch_size)

	test_y = model.predict_classes(out)

	index=np.arange(len(test_y))+1
	header=np.array([0,0], dtype='int32')
	final_out=np.vstack((index,test_y))

	out_with_header = np.vstack((header,final_out.T))

	write(out_with_header,subfile)