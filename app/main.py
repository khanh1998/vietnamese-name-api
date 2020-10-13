from flask import Flask, jsonify, request
from transformers import RobertaForSequenceClassification
from app.utils import predict

model = RobertaForSequenceClassification.from_pretrained('vinai/phobert-base')
app = Flask(__name__)


@app.route('/')
def hello_world():
    return '/gender?names='


@app.route('/gender', methods=['POST'])
def segmentation():
    params = request.args.get('names')
    names = params.split(',')
    if names:
        result = predict(names, model)
        return jsonify({'result': result.tolist()})

