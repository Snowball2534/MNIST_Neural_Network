import numpy as np
'''
Simply train and test the logistic regression which would be one layer 
in the neural network model.
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

# train the model by using Gradient Descent
m = x.shape[1]

num_epochs = 2000
alpha = 0.1

large_num = 1e8
epsilon = 1e-6
thresh = 1e-4

w = np.random.rand(m)
b = np.random.rand()

c = np.zeros(num_epochs)

for epoch in range(num_epochs):
    # use larger learning rate at the beginning
    if epoch < 600:
        alpha = 0.1
    else:
        alpha = 0.01

    # forward Propagation
    a = x @ w + b
    a = 1 / (1 + np.exp(-a))
    # back Propagation
    w -= alpha * (x.T) @ (a - y)
    b -= alpha * (a - y).sum()

    # compute the cost
    cost = np.zeros(len(y))
    idx = (y == 0) & (a > 1 - thresh) | (y == 1) & (a < thresh)
    cost[idx] = large_num

    a[a < thresh] = thresh
    a[a > 1 - thresh] = thresh

    inv_idx = np.invert(idx)
    cost[inv_idx] = - y[inv_idx] * np.log(a[inv_idx]) - (1 - y[inv_idx]) * np.log(1 - a[inv_idx])
    c[epoch] = cost.sum()

    if epoch % 5 == 0:
        print('epoch = ', epoch + 1, 'cost = ', c[epoch])

    if epoch > 0 and abs(c[epoch - 1] - c[epoch]) < epsilon:
        break

# load test dataset
new_test = np.loadtxt('test.txt', delimiter=',')
print('Data loading done')
new_x = new_test / 255.0

# apply the model on the test dataset
a = new_x@w + b
a = 1 / (1 + np.exp(-a))
a[a >= 0.5] = 1
a[a < 0.5] = 0
np.savetxt("predicted_value.txt", np.reshape(a, (1,200)), fmt='%d', delimiter=',') # Save the results
print('Results saving done')
