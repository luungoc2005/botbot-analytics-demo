from routes.common_routes import common_routes_blueprint

@common_routes_blueprint.route('/')
def index():
    return f'Server is up and running'