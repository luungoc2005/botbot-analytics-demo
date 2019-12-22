from config import DATA_DIR
from routes.training_routes import training_routes_blueprint
from flask import jsonify
from os import listdir

@training_routes_blueprint.route('/demo_training_list')
def demo_training_list():
    items = listdir(DATA_DIR)
    return jsonify([ item for item in items if item.lower()[-4:] == 'json' ])