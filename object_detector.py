import tensorflow as tf
from PIL import Image
import numpy as np
from load_model import load_model

# tomato leaf dataset classes
classes = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
    ]



def object_detector(model, image_np_array) -> str:
    """

    """

    prediction = np.argmax(model.predict(image_np_array.reshape(-1,256,256,3)/255))



    return classes[prediction]

   
if __name__ == "__main__":
    model = load_model()
    image = Image.open('./img.jfif')
    image_np_array = np.asarray(image)
    print(object_detector(model, image_np_array))
