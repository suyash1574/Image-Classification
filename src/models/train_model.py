import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import mlflow.tensorflow
import numpy as np
import os

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

def train_model():
    (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = build_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    with mlflow.start_run():
        history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
        test_loss, test_acc = model.evaluate(x_test, y_test)
        mlflow.log_param("epochs", 10)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.tensorflow.log_model(model, "model")
        model.save("models/image_classifier.h5")

if __name__ == "__main__":
    from src.data.make_dataset import load_cifar10_data
    train_model()