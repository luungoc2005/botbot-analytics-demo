from flask import Blueprint

training_routes_blueprint = Blueprint('training', __name__)

from routes.training_routes import demo_training_list
from routes.training_routes import training_stats