from ..app import app
from flask import jsonify
from os import listdir

@app.route('/demo_list')
def demo_list():
    items = listdir('./data')
    return jsonify([ item for item in items if item.lower()[-3:] == 'csv' ])