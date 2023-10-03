import h5py
import numpy as np


def save_data(ds_name: str, x_data, y_data):
    save_path = "/Users/utilisateur/PycharmProjects/ImageTesto/datasets/"

    with h5py.File(name=f"{save_path}{ds_name}.hdf5", mode="w") as f_h5:
        f_h5.create_dataset(name="x_data", data=x_data)
        f_h5.create_dataset(name="y_data", data=y_data)


def load_data():
    ds_path = "/Users/utilisateur/PycharmProjects/ImageTesto/datasets/"

    train_data = h5py.File(name=f"{ds_path}train_data.hdf5", mode="r")
    var_x = np.array(train_data["x_data"][:])
    var_y = np.array(train_data["y_data"][:])

    return var_x, var_y
