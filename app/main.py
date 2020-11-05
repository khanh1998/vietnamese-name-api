from flask import Flask, jsonify, request
from app.Model import LSTM_classifier
from app.utils import test_model
import torch
import os

root_dir = os.getcwd()
model_folder = f'{root_dir}/app/static/rnn-character-level'
print(model_folder)
cpu = torch.device('cpu')

app = Flask(__name__)
model = LSTM_classifier(hidden_size=256)
model.load_state_dict(torch.load(model_folder, map_location=cpu))

@app.route('/')
def hello_world():
    return '/gender?names='


@app.route('/gender', methods=['POST'])
def segmentation():
    params = request.args.get('names')
    names = params.split(',')
    if names:
        result = test_model(names, model, cpu)
        return jsonify({'result': result})
    return jsonify({'result': []})


