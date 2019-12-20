from flask import Blueprint

history_routes_blueprint = Blueprint('history', __name__)

import routes.history_routes.demo_list
import routes.history_routes.clustering
import routes.history_routes.intents_list
import routes.history_routes.top_intents
import routes.history_routes.top_words
import routes.history_routes.words_trend
import routes.history_routes.top_similar_words
import routes.history_routes.intents_trend
