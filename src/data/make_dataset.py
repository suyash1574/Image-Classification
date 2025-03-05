import tensorflow as tf
import numpy as np
import os

def load_cifar10_data(data_dir="data/raw"):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    os.makedirs(data_dir, exist_ok=True)
    np.save(f"{data_dir}/x_train.npy", x_train)
    np.save(f"{data_dir}/y_train.npy", y_train)
    np.save(f"{data_dir}/x_test.npy", x_test)
    np.save(f"{data_dir}/y_test.npy", y_test)
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    load_cifar10_data()