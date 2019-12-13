import csv
import numpy as np
from numpy import genfromtxt

def read(filename):
	my_data = genfromtxt(filename, delimiter=',')
	return my_data[1:]

def write(array,filename):
	np.savetxt(filename,array,delimiter=',',fmt='%d')

def data_label_split(array):
	return array[:,1:], array[:,0]