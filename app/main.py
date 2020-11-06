from flask import Flask, jsonify, request

app = Flask(__name__)
from app.utils import test_model

@app.route('/')
def hello_world():
    return '/gender?names='


@app.route('/gender', methods=['POST'])
def segmentation():
    params = request.args.get('names')
    names = params.split(',')
    if names:
        result = test_model(names)
        return jsonify({'result': result})
    return jsonify({'result': []})


