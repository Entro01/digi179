# https://github.com/imsanjoykb/Brain-Tumor-Detection-CNN-Architecture

# Importing necessary libraries
import os
import numpy as np

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow_hub as hub

# Define a flask app
app = Flask(__name__)

# Load the model
model = load_model('brain_tumor_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})

def predict_with_model(img_path, trained_model):
   test_img = image.load_img(img_path, target_size = (224,224))
   test_img = image.img_to_array(test_img)
   test_img = test_img / 255
   test_img = np.expand_dims(test_img, axis=0)
   prediction = trained_model.predict(test_img)

   if prediction <= 0.5:
       return "The person does not have a brain tumor"
   else:
       return "The person has a brain tumor"

@app.route('/', methods=['GET'])
def homepage():
   # Main page
   return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
   if request.method == 'POST':
       # Get the file from post request
       uploaded_file = request.files['file']

       # Save the file to ./uploads
       base_path = os.path.dirname(__file__)
       file_path = os.path.join(base_path, 'uploads', secure_filename(uploaded_file.filename))
       uploaded_file.save(file_path)

       # Make prediction
       prediction = predict_with_model(file_path, model)
       return prediction
   return None

if __name__ == '__main__':
   app.run(debug=False)
