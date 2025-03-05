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
        layers.Dropout(0.5),  # Add dropout to reduce overfitting
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
        history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))  # Increase to 20 epochs
        test_loss, test_acc = model.evaluate(x_test, y_test)
        mlflow.log_param("epochs", 20)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.tensorflow.log_model(model, "model")
        model.save("models/image_classifier.h5")

def test_model(model, x_test, y_test):
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    predictions = model.predict(x_test[:5])
    for i, pred in enumerate(predictions):
        pred_class = class_names[np.argmax(pred)]
        true_class = class_names[y_test[i][0]]
        print(f"Predicted: {pred_class}, True: {true_class}, Confidence: {pred[np.argmax(pred)]:.4f}")

if __name__ == "__main__":
    from src.data.make_dataset import load_cifar10_data
    (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    train_model()
    model = tf.keras.models.load_model("models/image_classifier.h5")
    test_model(model, x_test / 255.0, y_test)