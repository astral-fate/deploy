from flask import Flask, render_template, request

import joblib

import numpy as np

app = Flask(__name__)

reg_model = joblib.load("Regression_model.pkl")

clf_model = joblib.load("classification_model.pkl")

@app.route('/')

def home():

    return render_template('index.html')

@app.route('/results', methods=['POST'])

def results():

    if request.method == 'POST':

        features = [float(x) for x in request.form.values()]

        features = np.array(features).reshape(1, -1)

        reg_pred = reg_model.predict(features)

        clf_pred = clf_model.predict(features)

        return render_template('index.html', reg_pred=reg_pred[0], clf_pred=clf_pred[0])

if __name__ == '__main__':

    app.run(debug=True)

