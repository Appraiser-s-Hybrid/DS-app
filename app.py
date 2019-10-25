# Import libraries

import numpy as np
from flask import Flask, request, jsonify
import pickle
from xgboost import XGBRegressor
import pandas as pd
#from flask_cors import CORS, cross_origin


# load model
model = pickle.load(open('model.pkl','rb'))

# app
app = Flask(__name__)
#cors = CORS(app)
#app.config['CORS_HEADERS'] = 'Content-Type'

# routes
#@app.route('/api', methods=['POST'])
#@cross_origin(origin='appraisely.herokuapp.com',headers=['Content- Type','Authorization'])
#def helloWorld():
#  return "Hello, cross-origin-world!"
@app.route('/api', methods=['POST'])
def predict():

    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = model.predict(data_df)

    # send back to browser
    output = {'results': float(result)}
    response = jsonify(results=output)
    response.headers.add('Access-Control-Allow-Origin', '*')

    # return data
    return response


if __name__ == '__main__':
    app.run()
