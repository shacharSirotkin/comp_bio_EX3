import numpy as np
import sys

OUTPUT_FILE = 'w0.npz'

# sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# sigmoid derivative
def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))


def fprop(x, params):
    # fprop for one example
    w1, b1, w2, b2, w3, b3 = [params[key] for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')]
    z1 = np.dot(w1, x) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(w2, h1) + b2
    h2 = sigmoid(z2)
    z3 = np.dot(w3, h2) + b3
    h3 = sigmoid(z3)
    ret = {'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'z3': z3, 'h3': h3}
    for key in params:
        ret[key] = params[key]
    return ret


# bprop for one example
def bprop(x, y, fprop_cache):
    w2, w3, b1, b2, b3, z1, h1, z2, h2, z3, h3 = [fprop_cache[key] for key in
                                                  ('W2', 'W3', 'b1', 'b2', 'b3',
                                                   'z1', 'h1', 'z2', 'h2', 'z3', 'h3')]
    dl = h3 - y
    dw3 = dl.dot(h2.T)
    dh2 = dl.T.dot(W3)
    dg2 = dh2.T * sigmoid_derivative(z2)
    dw2 = dg2.dot(h1.T)
    dh1 = dg2.T.dot(W2)
    dg1 = dh1.T * sigmoid_derivative(z1)
    dw1 = dg1.dot(x.T)
    return {'db1': dg1, 'dW1': dw1, 'db2': dg2, 'dW2': dw2, 'db3': dl, 'dW3': dw3}


if __name__ == '__main__':
    # read training set and test set
    file_1 = sys.argv[1]
    file_2 = sys.argv[2]
    lines = None
    with open(file_1, 'r') as test_file:
        lines = test_file.readlines()
    lines_x = [line.split('   ')[0] for line in lines]
    lines_x = [map(int, line) for line in lines_x]
    lines_y = [int(line.split('   ')[1].rstrip('\n')) for line in lines]
    data = list(zip(lines_x, lines_y))
    np.random.shuffle(data)
    train_x, train_y = zip(*data)
    train_x, train_y = np.array(train_x), np.array(train_y)

    print "==================finish loading=============="

    lines = None
    with open(file_2, 'r') as test_file:
        lines = test_file.readlines()
    lines_x = [line.split('   ')[0] for line in lines]
    lines_x = [map(int, line) for line in lines_x]
    lines_y = [int(line.split('   ')[1].rstrip('\n')) for line in lines]
    data = list(zip(lines_x, lines_y))
    np.random.shuffle(data)
    test_x, test_y = zip(*data)
    test_x, test_y = np.array(test_x), np.array(test_y)

    # Initialize random parameters and inputs
    W1 = np.random.uniform(-1, 1, size=(25, 16))
    b1 = np.random.uniform(-1, 1, size=(25, 1))
    W2 = np.random.uniform(-1, 1, size=(10, 25))
    b2 = np.random.uniform(-1, 1, size=(10, 1))
    W3 = np.random.uniform(-1, 1, size=(1, 10))
    b3 = np.random.uniform(-1, 1, size=(1, 1))
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

    epochs = 30
    learning_rate = 0.3
    # run all epochs
    accuracy_train = []
    accuracy_valid = []
    for i in range(epochs):
        params2 = None
        print "epoch", i
        # run one epoch
        for x, y in zip(train_x, train_y):
            x = np.reshape(x, (16, 1))
            fprop_cache = fprop(x, params)
            bprop_cache = bprop(x, y, fprop_cache)
            params = {'W1': params['W1'] - learning_rate * bprop_cache['dW1'],
                      'b1': params['b1'] - learning_rate * bprop_cache['db1'],
                      'W2': params['W2'] - learning_rate * bprop_cache['dW2'],
                      'b2': params['b2'] - learning_rate * bprop_cache['db2'],
                      'W3': params['W3'] - learning_rate * bprop_cache['dW3'],
                      'b3': params['b3'] - learning_rate * bprop_cache['db3']}

        successCounter = 0
        for x, y in zip(train_x, train_y):
            x = np.reshape(x, (16, 1))
            fprop_cache = fprop(x, params)
            if int((fprop_cache['h3'] > 0.5)) == y:
                successCounter += 1

        print successCounter
        print 'train  accuracy', successCounter / float(len(train_x))
        accuracy_train.append(successCounter / float(len(train_x)))

        successCounter = 0
        for x, y in zip(test_x, test_y):
            x = np.reshape(x, (16, 1))
            fprop_cache = fprop(x, params)
            if int((fprop_cache['h3'] > 0.5)) == y:
                successCounter += 1

        print successCounter
        print 'validation accuracy', successCounter/float(len(test_x))
        accuracy_valid.append(successCounter / float(len(test_x)))

        print "===============finish one epoch===================="

    # save the weights
    np.savez(OUTPUT_FILE, W1=params['W1'], b1=params['b1'], W2=params['W2'], b2=params['b2'],
             W3=params['W3'], b3=params['b3'])
