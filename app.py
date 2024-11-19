# importing necessary libraries and functions
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__) #Initialize the flask App
model = pickle.load(open('model', 'rb')) # loading the trained model

@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    init_news = [x for x in request.form.values()]
    

    prediction = model.predict(init_news) # making prediction
    if prediction==0:
        prediction="Real"
    else:
        prediction="Fake"


    return render_template('index.html', prediction_text='The News is Predicted as : {}'.format(prediction)) # rendering the predicted result

if __name__ == "__main__":
    app.run(debug=True)
