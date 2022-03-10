import imp
from tensorflow import keras

__model = None

def load_model(path = './mobileNetV2.h5'):
    global __model
    if not __model:
        __model = keras.models.load_model(path)
        
    return __model