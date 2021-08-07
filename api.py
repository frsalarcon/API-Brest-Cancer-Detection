# --------------------------------- paquetes --------------------------------- #

from typing import AnyStr
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from glob import glob
import os
from keras.preprocessing import image
from keras.datasets import mnist
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import uvicorn
from io import BytesIO
from PIL import Image

# ------------------------------------ API ----------------------------------- #

app = FastAPI(title='Hello world')

@app.get('/index')
async def hello_world():
    return "hello world"

# ---------------------------------- modelo ---------------------------------- #
modelo='model.h5'
pesos_modelo='pesos.h5'
model = load_model(modelo)
model.load_weights(pesos_modelo)


    
 # ------------------------------------ API ----------------------------------- #
def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    image = load_image_into_numpy_array(await file.read())
    image =np.resize(image,(100,100,3))
    x = np.expand_dims(image, axis=0)
    array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
        answer="Bening "
    elif answer == 1:
        answer="Maligno"
    st=str(answer)
    # # return answer
    return{"Predicción":st}



