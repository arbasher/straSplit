# Import all required libraries
import tensorflow as tf

tf.__version__
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime

# Split MNIST Train and Test data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Divide train set by 255
x_train, x_test = x_train.astype("float32") / 255, x_test.astype("float32") / 255
x_train, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)
# convert class vectors to binary class matrices
y_train, y_test = keras.utils.to_categorical(y_train, num_classes=10), keras.utils.to_categorical(y_test,
                                                                                                  num_classes=10)
# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(1, 1)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ]
)
# Model summary and Evaluation
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
start = datetime.now()
model.fit(x_train, y_train, batch_size=128, epochs=30, validation_split=0.1)
stop = datetime.now()
print("Time taken to execute:" + str(stop - start))
