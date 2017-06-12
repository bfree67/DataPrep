
import numpy as np
from pandas import read_csv
import copy

# load the dataset
dataframe = read_csv('sequence.csv', header = 0, engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
n,p = np.shape(dataset)

delay = 3

a = np.zeros((n,p*delay))

dataX = copy.copy(dataset[0:(n-delay),:])

for i in range(1,delay):
    a = dataset[i:(n-delay+i), :]
    dataX = np.concatenate((dataX, a), axis = 1)

dataXre=np.reshape(dataX,((n-delay),delay,p)) 