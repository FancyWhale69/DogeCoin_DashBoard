import json
import plotly
import numpy as np
import pandas as pd
from tensorflow import keras

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Line
import joblib



app = Flask(__name__)



# load data
date = pd.read_csv('../data/date.csv')
test = pd.read_csv('../data/test_data.csv')
full = pd.read_csv('../data/full_data.csv')
loss= pd.read_csv('../data/loss.csv')


# load model
model = keras.models.load_model('../models/Doge_Model.h5')
scaler = joblib.load('../models/scaler.gz')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Line(
                    x=date['Date'],
                    y=full['true']
                )
            ],

            'layout': {
                'title': 'DogeCoin Price',
                'yaxis': {
                    'title': "Price in USD"
                },
                'xaxis': {
                    'title': "Date"
                }
            }
        },
        {
            'data': [
                Line(
                    y=loss['loss'],
                    name="Training_loss"
                ),
                Line(
                    y=loss['val_loss'],
                    name="val_loss"
                )
            ],

            'layout': {
                'title': 'Training_loss VS val_loss',
                'yaxis': {
                    'title': "Value"
                },
                'xaxis': {
                    'title': "Epochs"
                }
            }
        },
        {
            'data': [
                Line(
                    y=test['test'],
                    name="Test_data"
                ),
                Line(
                    y=test['pred'],
                    name="Pred_data"
                )
            ],

            'layout': {
                'title': 'Test_data VS Pred_data',
                'yaxis': {
                    'title': "Price in USD"
                },
                'xaxis': {
                    'title': "Index"
                }
            }
        },
        {
            'data': [
                Line(
                    x=date['Date'],
                    y=full['true'],
                    name="Actual_data"
                ),
                Line(
                    x=date['Date'],
                    y=full['pred'],
                    name="Pred_data"
                )
            ],

            'layout': {
                'title': 'Actual_data VS Pred_data',
                'yaxis': {
                    'title': "Price in USD"
                },
                'xaxis': {
                    'title': "Date"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    query = query.split(',')
    p=float(query[0].strip())
    c=float(query[1].strip())
    v=float(query[2].strip())
    point= scaler.transform(np.array([[p,c,v]]))

    # use model to predict classification for query
    classification_results = model.predict(point.reshape(1,1,3))[0][0]
    

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(port=3001)


if __name__ == '__main__':
    main()