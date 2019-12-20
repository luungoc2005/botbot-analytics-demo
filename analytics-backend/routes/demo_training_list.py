from main_app import app, DATA_DIR
from flask import jsonify
from os import listdir

@app.route('/demo_training_list')
def demo_training_list():
    items = listdir(DATA_DIR)
    return jsonify([ item for item in items if item.lower()[-4:] == 'json' ])