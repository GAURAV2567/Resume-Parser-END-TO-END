import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features =  request.form.values()
    docs=model.make_doc(int_features)
    
    lst=[]         
    for entity in doc.ents:
        #print((entity.text,entity.label_))
        lst.append((entity.text,entity.label_))

    output = lst

    return render_template('index.html', prediction_text='OUTPUT:- {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
