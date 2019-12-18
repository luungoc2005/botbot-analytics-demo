from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from .routes import index, demo_list