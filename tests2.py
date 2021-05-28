import mnist_loader
import ANN2

import numpy as np

debugdata = [(np.array([[121 / 225], [146 / 225]]), np.array([[0], [1]])),
             (np.array([[78 / 225], [221 / 225]]), np.array([[1], [0]])),
             (np.array([[22 / 225], [166 / 225]]), np.array([[0], [1]]))]

#
# print(training_data[0][1].shape)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = ANN2.Network([2, 2, 2])
net.SGD(10, 3, 1, debugdata)  ##  batchSize, eta, epochs,
# print(net.eval(test_data))
