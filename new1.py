import numpy as np
import sys
from io import StringIO

OUTPUT_FILE = 'predictions.txt'


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fprop(x, params):
    # Follows procedure given in notes
    w1, b1, w2, b2, w3, b3 = [params[key] for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3')]
    z1 = np.dot(w1, x) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(w2, h1) + b2
    h2 = sigmoid(z2)
    z3 = np.dot(w3, h2) + b3
    h3 = sigmoid(z3)
    return h3


validation_file = sys.argv[1]
w_file = sys.argv[2]
validation_lines = [line.rstrip('\n') for line in open(validation_file)]
validation_inputs = [map(int,line ) for line in validation_lines]
validation_size = len(validation_lines)

weights = np.load(w_file)
W1 = weights['W1']
b1 = weights['b1']
W2 = weights['W2']
b2 = weights['b2']
W3 = weights['W3']
b3 = weights['b3']
params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

loss = 0
accuracy = 0
output_fd = open(OUTPUT_FILE, 'w')
for x in validation_inputs:
    x = np.reshape(x, (16, 1))
    pred = int((fprop(x,params)>0.5))
    output_fd.write(str(pred) + '\n')
output_fd.close()