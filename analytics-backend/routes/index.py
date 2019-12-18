from ..app import app

@app.route('/')
def index():
    return f'Server is up and running'