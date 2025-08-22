from operator import xor
import os 
import sys
import cv2 

import tensorflow as tf
from tensorflow import keras
import numpy as np

def AdjustRealData(file_path):
    for file in os.listdir(file_path):
        if file.endswith( (".png", ".jpeg", ".jpg") ):
            img = cv2.imread(os.path.join(file_path, file))
            if img.shape[0] != 28 or img.shape[1] != 28:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (28, 28))
                cv2.imwrite(os.path.join(file_path, file), img)    

def RecognizeRealData(interpreter, file_img):
    img = cv2.imread(file_img, 0)
    (h, w) = img.shape[:2]
    show = img

    img = cv2.resize(img, (28, 28))
    data = np.asarray(img)
    data = data.reshape(1, 28*28)
    data = data.astype("float32")
    data /= 255
    
    width = 400
    height = int( h*(width/float(w)) )

    show = cv2.resize(show, (width, height))
    cv2.imshow(f"{file_img}", show)

    print("\n[+] Press SPACE_BAR to continue: ")
    while True:
        if (cv2.waitKey(0) & 0xFF) == 32:
            cv2.destroyAllWindows()
            break    

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_details["index"], data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details["index"])[0]
    return output_data


def RecognizeMNISTData(interpreter, index):
    if index < 0:
        index = 1
    elif index > 10_000:
        index = 9_999    

    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    data = x_test[index].reshape(1, 28*28)
    data = data.astype("float32")
    data /= 255

    (h, w) = (28, 28)
    show = x_test[index]
    
    width = 400
    height = int( h*(width/float(w)) )

    show = cv2.resize(show, (width, height))
    cv2.imshow("First MNIST dataset", show)
    print("\n[+] Press SPACE_BAR to continue: ")
    while True:
        if (cv2.waitKey(0) & 0xFF) == 32:
            cv2.destroyAllWindows()
            break    

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_details["index"], data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details["index"])[0]
    return output_data

if __name__ == '__main__':

    interpreter = tf.lite.Interpreter(model_path="./OHE_models/TFLite_model.tflite")
    interpreter.allocate_tensors()
    interpreter_details = interpreter.get_input_details()[0]
    
    for key, value in interpreter_details.items():
        print(f"{key} {value}")
    print()

    
    """"
    dir_photo = "./OHE_images"

    while not os.path.exists(f"./{dir_photo}"):
        dir_photo = input("[+] Directory: ")
    
    AdjustRealData(dir_photo)
    for AdjImg in os.listdir(dir_photo):
        if AdjImg.endswith( (".png", ".jpeg", ".jpg") ):
            prediction = RecognizeRealData(interpreter, os.path.join(dir_photo, AdjImg))
            for x, out in enumerate(prediction):
                print(f"[+] Class: {x} \t Probability: {round(out)}",)
    """                  
    
    for x in range(1, 101):
        prediction = RecognizeMNISTData(interpreter, x)
        for x, out in enumerate(prediction):
            print(f"[+] Class: {x} \t Probability: {round(out)}",)

            
