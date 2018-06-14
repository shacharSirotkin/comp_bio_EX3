import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fprop(x, params):
    # Follows procedure given in notes
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    # print W1
    z1 = np.dot(W1, x) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = sigmoid(z2)
    out = h2[0][0]
    loss = -(y * np.log(h2) + (1-y) * np.log(1-h2))
    ret = {'z1': z1, 'h1': h1, 'out': out, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret


def bprop(x, y, fprop_cache):
    z1, h1_outs, net_output, W1, b1, W2, b2 = [fprop_cache[key] for key in ('z1', 'h1', 'out','W1', 'b1', 'W2', 'b2')]
    out_error = (y - net_output)
    output_correct = out_error * net_output * (1 - net_output)
    subs_w2 = np.ones(h1_outs.shape) - h1_outs
    w2_delta = np.multiply(subs_w2, h1_outs)
    w2_sigma = np.sum(W2 * output_correct)
    w2_update = w2_delta * w2_sigma
    db2 = out_error
    dz1 = np.dot(fprop_cache['W2'].T, out_error) * (h1_outs * (1 - h1_outs))
    db1 = dz1
    dW1 = np.dot(dz1, x.T)
    return {'db1': db1, 'dW1': dW1, 'db2': db2, 'dW2': w2_update}

    # self.o_error = y - o  # error in output
    # self.o_delta = self.o_error * self.sigmoidPrime(o)
    #
    # self.z2_error = self.o_delta.dot(self.W2.T)
    # self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
    #
    # self.W1 += X.T.dot(self.z2_delta)
    # self.W2 += self.z2.T.dot(self.o_delta)


if __name__ == '__main__':
    # read training set and test set
    lines = None
    with open('nn0.txt','r') as nn0_file:
        lines = nn0_file.readlines()
    lines_x = [line.split('   ')[0] for line in lines]
    lines_x = [map(int, line) for line in lines_x]
    lines_y = [int(line.split('   ')[1].rstrip('\n')) for line in lines]
    data = list(zip(lines_x, lines_y))
    np.random.shuffle(data)
    train_x, train_y = zip(*data)
    train_x, train_y = np.array(train_x), np.array(train_y)

    print "==================finish loading=============="

    # split training set to train and dev
    dev_size = int(round(train_x.size * 0.2)/train_x[0].size)
    dev_x, dev_y = train_x[-dev_size:, :], train_y[-dev_size:]
    train_x, train_y = train_x[:-dev_size, :], train_y[:-dev_size]

    # Initialize random parameters and inputs
    W1 = np.random.uniform(-1, 1, size=(25, 16))
    b1 = np.random.uniform(-1, 1, size=(25, 1))
    W2 = np.random.uniform(-1, 1, size=(1, 25))
    b2 = np.random.uniform(-1, 1, size=(1, 1))
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    epochs = 20
    learning_rate = 0.01
    # run all epochs
    for i in range(epochs):
        params2 = None
        print "epoch", i
        # run one epoch
        for x, y in zip(train_x, train_y):
            x = np.reshape(x, (16, 1))
            fprop_cache = fprop(x, params)
            bprop_cache = bprop(x, y, fprop_cache)
            print bprop_cache['dW1'][0][0]
            params = {'W1': params['W1'] - learning_rate * bprop_cache['dW1'],
                      'b1': params['b1'] - learning_rate * bprop_cache['db1'],
                      'W2': params['W2'] - learning_rate * bprop_cache['dW2'],
                      'b2': params['b2'] - learning_rate * bprop_cache['db2']}

        successCounter = 0
        for x, y in zip(dev_x, dev_y):
            x = np.reshape(x, (16, 1))
            fprop_cache = fprop(x, params)
            if np.argmax(fprop_cache['h2']) == y:
                successCounter += 1

        print successCounter
        print successCounter/float(dev_size)

        print "===============finish one epoch===================="

    with open("w0.txt", 'w') as f:
        for w in list(params['W1']):
            f.write(str(w) + "\n")
        f.write('W2:')
        for w in list(params['W2']):
            f.write(str(w) + "\n")