import numpy as np
import function as f
'''
 Train and test a neural network with one hidden layer. The number of hidden units 
 is the square root of the number of input units (here, the number of input 
 units is 784, so the number of hidden units is 28). The used activation 
 function is logistic in both layers.
'''


# load the data
def data_loader(file):
    a = np.genfromtxt(file, delimiter=',', skip_header=0)
    x = a[:, 1:] / 255.0
    y = a[:, 0]
    return (x, y)


x_train, y_train = data_loader('mnist_train.csv')
print('Data loading done')

# note that only use the number 0 (label 0) and 8 (label 1) to train and test this layer
test_labels = [0, 8]
indices = np.where(np.isin(y_train, test_labels))[0]

x = x_train[indices]
y = y_train[indices]

y[y == test_labels[0]] = 0
y[y == test_labels[1]] = 1

# train the neural network
num_hidden_units = 28
lr = 0.01
num_epochs = 200

W1, W2 = f.nnet(x, y, num_hidden_units, lr, num_epochs)

# test the neural network
test_x = np.loadtxt('test.txt', delimiter=',')
test_x = test_x / 255.0

f.test_nnet(test_x, W1, W2)
