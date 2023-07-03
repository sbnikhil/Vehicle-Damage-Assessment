import os.path
import numpy as np
import tensorflow as tf
from tensorflow import *
from flask import Flask,app, request, render_template
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image

app = Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/login.html')
def login():
    return render_template('login.html')

@app.route('/register.html')
def register():
    return render_template('register.html')

model1 = load_model(r'D:\Vehicle Damage\Model\body.h5')
model2 = load_model(r'D:\Vehicle Damage\Model\level.h5')

@app.route('/login_success')
def login_success():
    return render_template('predict.html')

@app.route('/predict', methods = ["GET","POST"])
def predict():

    value = ""
    if request.method == "POST":
        f = request.files['imageUpload']
        basepath = os.path.dirname(r"D:\Vehicle Damage")
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        img = image.load_img(filepath, target_size = (224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        img_data = preprocess_input(x)
        pred1 = np.argmax(model1.predict(img_data))
        pred2 = np.argmax(model2.predict(img_data))

        index1 = ['front','rear','side']
        index2 = ['minor','moderate','severe']

        res1 = index1[pred1]
        res2 = index2[pred2]
        
        if(res1 == "front" and res2=="minor"):
            value = "3000-6000 INR"

        elif(res1== "front" and res2 == "moderate"):
            value = "6500-9500 INR"

        elif(res1 == "rear" and res2 == "minor"):
            value = "4000 - 6000 INR"

        elif(res1 == "rear" and res2 == "moderate"):
            value = "8000-9000 INR"

        elif (res1 == "rear" and res2 == "severe"):
            value = "11000-13500 INR"

        elif (res1 == "side" and res2 == "minor"):
            value = "6000-8000 INR"

        elif (res1 == "side" and res2 == "moderate"):
            value = "10000-12000 INR"

        elif (res1 == "side" and res2 == "severe"):
            value = "12000-15000 INR"

        else:
            value = "16000 - 50000 INR"
             
    return value

if __name__ == '__main__':
    app.run(debug = True, use_reloader=True)