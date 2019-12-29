from config import DATA_DIR
from routes.training_routes import training_routes_blueprint
from common import cache, utils, ignore_lists, task_scheduler

from flask import jsonify, escape, request
from werkzeug.exceptions import BadRequest
from os import listdir, path

@training_routes_blueprint.route('/training_test_query')
def training_test_query():
    file_name = request.args.get("file", "")
    test_query = request.args.get("query", "").strip()

    if len(test_query) == 0:
        return BadRequest('No text query provided')

    sid = request.args.get("sid", "")
    
    file_path = path.join(DATA_DIR, escape(file_name))
    if not path.exists(file_path) or not path.isfile(file_path):
        return BadRequest('File not found')

    return jsonify({
        'task_id': task_scheduler.run_task('tasks.training_stats', [
            '--file_name', file_name,
            '--query', test_query,
        ], request.url_root, sid)
    })