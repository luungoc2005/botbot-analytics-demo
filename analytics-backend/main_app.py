from gevent import monkey
monkey.patch_all()

from flask import Flask, Blueprint, request
from flask_cors import CORS
from flask_socketio import SocketIO
from common.task_scheduler import assign_task_target

from config import LOGS_DIR, DATA_DIR

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*', message_queue='redis://')

from routes.common_routes import common_routes_blueprint
from routes.history_routes import history_routes_blueprint
from routes.training_routes import training_routes_blueprint

app.register_blueprint(common_routes_blueprint)
app.register_blueprint(history_routes_blueprint)
app.register_blueprint(training_routes_blueprint)

import json
clients = []
@socketio.on('connected')
def connected(message):
    global clients
    print("%s connected" % (request.sid))
    json_data = json.loads(message)

    for task_id in json_data:
        assign_task_target(task_id, request.sid)

    socketio.send(json.dumps({ 'hello': 'world' }), room=request.sid)

    clients.append(request.sid)
    
@socketio.on('disconnect')
def disconnect():
    global clients
    print("%s disconnected" % (request.sid))
    if request.sid in clients:
        clients.remove(request.sid)

if __name__ == '__main__':
    socketio.run(app)