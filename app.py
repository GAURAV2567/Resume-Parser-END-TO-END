import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    docs=model.pipe(int_features)
    lst=[]
    for doc in docs:
        
        for entity in doc.ents:
            #print((entity.text,entity.label_))
            lst.append((entity.text,entity.label_))

    output = lst

    return render_template('index.html', prediction_text='OUTPUT:- {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)