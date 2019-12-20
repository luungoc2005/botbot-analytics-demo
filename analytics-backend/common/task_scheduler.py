import uuid
import subprocess
# import shlex

from config import LOGS_DIR

from os import path

TASK_PROCESSES = {}
TASK_TARGETS = {}

def assign_task_target(task_id, sid):
    TASK_TARGETS[task_id] = sid

def run_task(task_name, args, url_root, sid):
    global TASK_PROCESSES
    global TASK_TARGETS

    task_id = str(uuid.uuid1())
    task_args = [
        'python3', '-m', task_name,
        '--sid', sid,
        '--callback_url', f'{url_root}internal/task_response?task_id={task_id}'
    ]
    task_args.extend(args)
    # task_args = [shlex.quote(s) for s in task_args]

    log_file_name = path.join(LOGS_DIR, task_id + '.log')

    TASK_TARGETS[task_id] = sid

    with open(log_file_name, 'w', encoding='utf8') as log_fp:
        TASK_PROCESSES[task_id] = subprocess.Popen(task_args, stdout=log_fp)

    return task_id