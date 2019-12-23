from config import DATA_DIR
from routes.training_routes import training_routes_blueprint
from common import cache, utils, ignore_lists, task_scheduler

from flask import jsonify, escape, request
from werkzeug.exceptions import BadRequest
from os import listdir, path

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

import json

@training_routes_blueprint.route('/training_stats')
def training_stats():
    file_name = request.args.get("file", "")

    sid = request.args.get("sid", "")
    
    file_path = path.join(DATA_DIR, escape(file_name))
    if not path.exists(file_path) or not path.isfile(file_path):
        return BadRequest('File not found')

    return jsonify({
        'task_id': task_scheduler.run_task('tasks.training_stats', [
            '--file_name', file_name,
        ], request.url_root, sid)
    })