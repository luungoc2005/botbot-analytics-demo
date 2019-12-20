from flask import Blueprint

common_routes_blueprint = Blueprint('common', __name__)

import routes.common_routes.index
import routes.common_routes.task_response