from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

DATA_DIR = './data'

from .routes import \
    index, \
    demo_list, \
    clustering, \
    intents_list, \
    top_intents, \
    top_words, \
    words_trend, \
    intents_trend