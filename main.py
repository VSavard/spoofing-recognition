from models.neurones import neuron_network
from methods.dataset_import_save import load_data

x_train, y_train = load_data()

print("x_train shape", x_train.shape)
print("y_train shape", y_train.shape)

x_train_reshape = x_train.reshape(x_train.shape[0], -1) / x_train.max()
x_train_reshape = x_train_reshape.T
y_train = y_train.reshape((1, y_train.shape[0]))

neuron_network(x_train_reshape, y_train, hidden_layers=(16, 16, 16), learning_rate=0.1, n_iter=3000)
