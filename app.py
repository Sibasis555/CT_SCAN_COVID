# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 22:34:20 2020

@author: Krish Naik
"""

from flask import Flask, request, jsonify, render_template,redirect
import pickle
import torch
#from torch.utils.data import DataLoader
from torchvision import datasets,transforms,models
import os
from werkzeug.utils import secure_filename
from PIL import Image
from werkzeug.exceptions import BadRequest

app = Flask(__name__)
#map_location=torch.device('cpu')
model= pickle.load(open('imageCovid.pkl', 'rb'))

def model_predict(img_path, model):
    
    print(img_path)
    transform = transforms.Compose([
		    transforms.Resize(256),
		    transforms.CenterCrop(224),
		    transforms.ToTensor(),
		    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        	])
    image = Image.open(img_path).convert('RGB')
    x = transform(image).unsqueeze(0)
    
    model.eval()

    with torch.no_grad():
         preds=model(x).argmax()

    if preds==0:
        preds="Covid positive"
    else:
        preds="Covid negative"
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        #if request.files:
        #f= request.files.get('file')
        f=request.files["file"]
        
        if not f:
         return BadRequest("File is not present in the request")
        if f.filename == '':
           return BadRequest("Filename is not present in the request")
        if not f.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
          return BadRequest("Invalid file type")

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Make ctction
        preds = model_predict(f, model)
        print("======================")
        print(preds)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)
