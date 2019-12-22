from main_app import DATA_DIR, socketio, app
from routes.common_routes import common_routes_blueprint

from common import cache, utils, vn_utils, ignore_lists, task_scheduler

import numpy as np

from flask import jsonify, escape, request
from werkzeug.exceptions import BadRequest
from os import listdir, path

# from flask_socketio import emit, send

import string
import json

@common_routes_blueprint.route('/internal/task_response', methods=['POST'])
def internal_task_cancel():
    task_id = request.args.get("task_id", "")

    if len(task_id) == 0:
        return BadRequest()

    # TODO: implement this