import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
import mlflow
import mlflow.tensorflow
import numpy as np
import os

def build_model():
    # Load VGG16 pre-trained on ImageNet, exclude top layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    base_model.trainable = False  # Freeze pre-trained layers

    # Define input and connect layers using Functional API
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = base_model(inputs, training=False)  # Pass input through VGG16
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Create model
    model = Model(inputs, outputs)
    return model

def train_model():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Build and compile model
    model = build_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train with MLflow tracking
    with mlflow.start_run():
        history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=64)
        test_loss, test_acc = model.evaluate(x_test, y_test)
        mlflow.log_param("epochs", 10)
        mlflow.log_param("batch_size", 64)
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