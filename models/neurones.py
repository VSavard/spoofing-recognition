from methods.training_methods import initialisation, forward_propagation, update, back_propagation, predict
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np


def neuron_network(var_x, var_y, hidden_layers=(32, 32, 32), learning_rate=0.01, n_iter=1000):

    dim = list(hidden_layers)
    dim.insert(0, var_x.shape[0])
    dim.append(var_y.shape[0])
    np.random.seed(1)
    parameters = initialisation(dim)

    training_history = np.zeros((int(n_iter), 2))

    var_c = len(parameters) // 2

    for i in tqdm(range(n_iter)):

        activations = forward_propagation(var_x=var_x, parameters=parameters)
        gradients = back_propagation(var_y=var_y, parameters=parameters, activations=activations)
        parameters = update(gradients=gradients, parameters=parameters, learning_rate=learning_rate)
        var_af = activations["var_a" + str(var_c)]

        training_history[i, 0] = (log_loss(var_y.flatten(), var_af.flatten()))
        y_pred = predict(var_x, parameters)
        training_history[i, 1] = (accuracy_score(var_y.flatten(), y_pred.flatten()))


    plt.style.use("dark_background")
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label="Train loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label="Train accuracy")
    plt.legend()
    plt.show()

    return training_history
