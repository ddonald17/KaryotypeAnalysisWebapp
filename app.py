from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
# Keras
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename

import cv2


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Adagrad4.9-saved-model-15-loss-0.17.hdf5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          

# Class labels
class_labels = ['Down Syndrome','Klinefelter Syndrome','Normal Female', 'Normal Male', 't(9;22)']
class_info = ['randomefhuweh', 'njfcenfniwe', 'wqnfuqwndiqw', 'wudnqwdiqwi', 'jsandjasnjnj']

def model_predict(img_path, model):
    #img = tf.keras.utils.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    #x =  np.asarray(img)
    #x = np.expand_dims(x, axis=0)
    img = cv2.resize(cv2.imread(img_path),(224,224))
    img_normalized = img/255
        
    preds = np.argmax(model.predict(np.array([img_normalized])))
    print(preds)
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #Sx = preprocess_input(x, mode='caffe')

    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for genus
        #pred_class = np.argmax(preds) 
        result = class_labels[preds]
        info = class_info[preds]
        return jsonify({'result': result, 'class_info': info})
    return None

if __name__ == '__main__':
    app.run(debug=True)

