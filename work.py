# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import sys

EPOCHS = 128
BATCH_SIZE = 64 
RESHAPED= 28*28

N_HIDDEN = 512
OHE_OUT = 10

VERBOSE = 1
VALIDATION_SPLIT = 0.2

DROPOUT = 0.5

def LoadData():
    mnist = keras.datasets.mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60_000, RESHAPED)
    x_test = x_test.reshape(10_000, RESHAPED)
    x_train = x_train.astype("float32")
    x_test  = x_test.astype("float32")
    x_train /= 255 
    x_test /= 255

    y_train = tf.keras.utils.to_categorical(y_train, OHE_OUT)
    y_test = tf.keras.utils.to_categorical(y_test, OHE_OUT) 

    return (x_train, y_train), (x_test, y_test)

def GetSize(file_path):
    return sys.getsizeof(file_path)
def ConvertBytes(size, unit=None):
    Conv = {
        "mb": [round(size/1024*1024, 3), "Megabytes"],
        "kb": [round(size/1024, 3), "Kilobytes"],
        "by": [size, "bytes"]
    }
    out = Conv.get(unit, "by")
    print(f"File size: {out[0]} {out[1]}") 




(x_train, y_train), (x_test, y_test) = LoadData()
print()
print(f"x_train sample: {0}", x_train.shape[0])
print(f"x_train np.size: {0}", x_train.shape[1])
print(f"x_test sample: {0}", x_test.shape[0])
print(f"x_test np.size: {0}", x_test.shape[1])
print()

model = keras.models.Sequential()
model.add( keras.layers.Dense(N_HIDDEN, input_shape=(RESHAPED,), activation="relu", name="dense_layer_1") )
model.add( keras.layers.Dropout(DROPOUT) )
model.add( keras.layers.Dense(N_HIDDEN, activation="relu", name="dense_layer_2") )
model.add( keras.layers.Dropout(DROPOUT) )
model.add( keras.layers.Dense(OHE_OUT, activation="softmax", name="dense_layer_3") )

model.summary()

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics= ["accuracy"])

TensorBoard = tf.keras.callbacks.TensorBoard(log_dir="./OHE_logs")
history = model.fit(x_train, y_train, 
          batch_size=BATCH_SIZE, epochs=EPOCHS, 
          verbose=VERBOSE, validation_data=(x_test, y_test), 
          validation_split=VALIDATION_SPLIT, callbacks=[TensorBoard])

evaluate = model.evaluate(x_test, y_test)
print(f"Test Loss: {evaluate[0]}")
print(f"Test Accuracy: {evaluate[1]}")

model.save("./OHE_models/keras_model.h5")
size = GetSize("./OHE_models/keras_model.h5")
ConvertBytes(size, "kb")

tf_lite_conv = tf.lite.TFLiteConverter.from_keras_model(model)
tf_lite_conv.optimization = [tf.lite.Optimize.DEFAULT]
tf_lite_conv.target_spec.supported_types = [tf.float32]
tf_lite_model = tf_lite_conv.convert()

open("./OHE_models/TFLite_model.tflite", "wb").write(tf_lite_model)
size = GetSize("./OHE_models/TFLite_model.tflite")
ConvertBytes(size, "kb")

print()
print("[+] Data Visualization: ")

loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, "y", label="Training Loss")
plt.plot(epochs, val_loss, "r", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
fig_1 = plt.gcf()
plt.show()
fig_1.savefig('./OHE_data/Training_and_Validation_Loss.png')
fig_1.savefig('./OHE_data/Training_and_Validation_Loss.pdf')


acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
plt.plot(epochs, acc, "y", label="Training Accuracy")
plt.plot(epochs, val_acc, "r", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
fig_2 = plt.gcf()
plt.show()
fig_2.savefig('./OHE_data/Training_and_Validation_Accuracy.png')
fig_2.savefig('./OHE_data/Training_and_Validation_Accuracy.pdf')

# -> See logs using:
# -> python -m tensorboard.main --logdir=OHE_logs/