import numpy as np


def initialisation(dim):
    parameters = {}
    var_c = len(dim)

    for c in range(1, var_c):
        parameters["var_w" + str(c)] = np.random.randn(dim[c], dim[c - 1])
        parameters["var_b" + str(c)] = np.random.randn(dim[c], 1)

    return parameters


def forward_propagation(var_x, parameters):
    activations = {"var_a0": var_x}
    var_c = len(parameters) // 2

    for c in range(1, var_c + 1):
        var_z = parameters["var_w" + str(c)].dot(activations["var_a" + str(c - 1)]) + parameters["var_b" + str(c)]
        activations["var_a" + str(c)] = 1 / (1 + np.exp(-var_z))

    return activations


def back_propagation(var_y, parameters, activations):

    m = var_y.shape[1]
    var_c = len(parameters) // 2

    var_dz = activations["var_a" + str(var_c)] - var_y
    gradients = {}

    for c in reversed(range(1, var_c + 1)):
        gradients["var_dw" + str(c)] = 1 / m * np.dot(var_dz, activations["var_a" + str(c -1)].T)
        gradients["var_db" + str(c)] = 1 / m * np.sum(var_dz, axis=1, keepdims=True)
        if c > 1:
            var_dz = np.dot(parameters["var_w" + str(c)].T, var_dz) * activations["var_a" + str(c - 1)] * \
                     (1 - activations["var_a" + str(c - 1)])

    return gradients


def update(gradients, parameters, learning_rate):
    var_c = len(parameters) // 2

    for c in range(1, var_c + 1):
        parameters["var_w" + str(c)] = parameters["var_w" + str(c)] - learning_rate * gradients["var_dw" + str(c)]
        parameters["var_b" + str(c)] = parameters["var_b" + str(c)] - learning_rate * gradients["var_db" + str(c)]

    return parameters


def predict(var_x, parameters):
    activations = forward_propagation(var_x, parameters)
    var_c = len(parameters) // 2

    var_af = activations["var_a" + str(var_c)]
    return var_af >= 0.5
