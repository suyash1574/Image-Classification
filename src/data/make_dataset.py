import tensorflow as tf
import numpy as np
import os
from PIL import Image

def load_cifar10_data(data_dir="data/raw"):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    os.makedirs(data_dir, exist_ok=True)
    np.save(f"{data_dir}/x_train.npy", x_train)
    np.save(f"{data_dir}/y_train.npy", y_train)
    np.save(f"{data_dir}/x_test.npy", x_test)
    np.save(f"{data_dir}/y_test.npy", y_test)
    return (x_train, y_train), (x_test, y_test)

def save_cat_image():
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    cat_idx = [i for i, y in enumerate(y_test) if y[0] == 3][0]  # 3 = cat
    img = Image.fromarray(x_test[cat_idx])
    img.save("cat_test.png")

if __name__ == "__main__":
    load_cifar10_data()
    save_cat_image()