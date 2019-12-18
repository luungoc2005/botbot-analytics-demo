from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DATA_DIR = './data'

from .routes import index, demo_list, clustering