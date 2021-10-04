import numpy as np
'''
 Training and testing functions.
'''


# sigmoid function
def logsig(_x):
    return 1 / (1 + np.exp(-_x))
# the derivative of sigmoid function
def diff_logsig(x):
    return logsig(x) * (1 - logsig(x))


def nnet(train_x, train_y, num_hidden_units, lr, num_epochs):
    '''
    Train the neural network by using Gradient Descent.

    Arg:
       train_x: The feature of training set.
       train_y: The label of training set.
       num_hidden_units: The number of units in the hidden layer.
       lr: Learning rate.
       num_epochs: The number of training epochs.

    Returns:
        wih.T: The weights of hidden layer.
        who: The weights of output layer.
    '''
    # prepare the input training data
    num_train = len(train_y)
    train_x = np.hstack((train_x, np.ones(num_train).reshape(-1, 1)))

    # configure the architecture of neural network
    num_input_units = train_x.shape[1]
    wih = np.random.uniform(low=-1, high=1, size=(num_hidden_units, num_input_units))
    who = np.random.uniform(low=-1, high=1, size=(1, num_hidden_units + 1))

    # train the neural network
    for epoch in range(1, num_epochs + 1):
        # prepare the output array
        out_o = np.zeros(num_train)
        out_h = np.zeros((num_train, num_hidden_units + 1))
        out_h[:, -1] = 1
        for ind in range(num_train):
            # forward Propagation
            row = train_x[ind]
            out_h[ind, :-1] = logsig(np.matmul(wih, row))
            out_o[ind] = 1 / (1 + np.exp(-sum(out_h[ind] @ who.T)))
            # backward Propagation
            delta = np.multiply(diff_logsig(out_h[ind]), (train_y[ind] - out_o[ind]) * np.squeeze(who))
            wih += lr * np.matmul(np.expand_dims(delta[:-1], axis=1), np.expand_dims(row, axis=0))
            who += np.expand_dims(lr * (train_y[ind] - out_o[ind]) * out_h[ind, :], axis=0)

        # compute the error
        error = sum(- train_y * np.log(out_o) - (1 - train_y) * np.log(1 - out_o))
        num_correct = sum((out_o > 0.5).astype(int) == train_y)

        print('epoch = ', epoch, ' error = {:.7}'.format(error),
              'correctly classified = {:.4%}'.format(num_correct / num_train))

    return wih.T, who


def test_nnet(test_x, wih, who):
    '''
    Test the neural network and save the results.

    Arg:
       test_x: The feature of testing set.
       wih.T: The weights of hidden layer.
       who: The weights of output layer.
    '''
    # prepare testing data
    test_x = np.concatenate((test_x, np.ones((len(test_x), 1))), axis=1)
    # first layer
    a1 = logsig(test_x@wih)
    # second layer
    a1 = np.concatenate((a1, np.ones((len(test_x),1))), axis=1)
    a2 = logsig(a1@(who.T))
    a2[a2 >= 0.5] = 1
    a2[a2 < 0.5] = 0

    # save the results
    np.savetxt("predicted_value.txt", np.reshape(a2, (1,len(test_x))), fmt='%d', delimiter=',')
    print('Results saving done')