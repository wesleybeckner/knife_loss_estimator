import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import joblib

from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
def home_check():
    return '''<h1>You have accessed the knife loss estimator</h1>'''


@app.route('/type-treatment/<string:treat>/gsm/<int:gsm>/'
           'width-11/<string:width11>/width-13/<int:width13>/'
           'slitwidth/<int:slitwidth>', methods=['GET'])
def schedule_api(treat, gsm, width11, width13, slitwidth):
    knn = joblib.load("data/{}.sav".format(treat))
    pred = knn.predict([[gsm, width11, width13, slitwidth]])
    pred = round(pred[0])
    return """{}""".format(pred)

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # run app in debug mode on port 5000
