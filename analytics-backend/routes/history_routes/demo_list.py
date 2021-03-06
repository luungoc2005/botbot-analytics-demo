from config import DATA_DIR
from routes.history_routes import history_routes_blueprint
from flask import jsonify
from os import listdir

@history_routes_blueprint.route('/demo_list')
def demo_list():
    items = listdir(DATA_DIR)
    return jsonify([
        {
            'name': item
        } 
        for item in items if item.lower()[-4:] == '.csv'
    ])