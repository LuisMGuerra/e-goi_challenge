# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 00:49:42 2019

@author: luis-
"""

from flask import Flask, request#, jsonify
import pandas as pd
import pickle

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)

@app.route('/', methods=['POST'])
def serve_prediction():
    data = request.get_json()
    df = pd.read_json(data)
    model = pickle.load(open("final_model.sav", 'rb'))
    predictions = pd.DataFrame(data = model.predict(df), columns=['LABEL'])
    return predictions.to_json(orient='records')

if __name__ == '__main__':
    app.run(debug=True)