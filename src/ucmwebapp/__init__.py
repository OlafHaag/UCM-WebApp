import os
from pathlib import Path

import dash
from flask import Flask
from flask_migrate import Migrate

from dotenv import load_dotenv

from src.ucmwebapp.exceptions import ImproperlyConfigured
from src.ucmwebapp.models import db

DOTENV_PATH = Path(__file__).parents[1] / '.env'
load_dotenv(DOTENV_PATH)

#################
# 1. App Config #
#################
if "DYNO" in os.environ:
    # the app is on Heroku
    debug = False
else:
    debug = True
    dotenv_path = Path(__file__).parents[1] / '.env'
    load_dotenv(dotenv_path)

# ToDo: First try with offline graphs?
#try:
#    py.sign_in(os.environ["PLOTLY_USERNAME"], os.environ["PLOTLY_API_KEY"])
#except KeyError:
#    raise ImproperlyConfigured("Plotly credentials not set in .env")

app_name = "UCMAnalysisDashApp"
server = Flask(app_name)

try:
    server.secret_key = os.environ["SECRET_KEY"]
except KeyError:
    raise ImproperlyConfigured("SECRET KEY not set in .env:")

external_js = []

external_stylesheets = [
    # dash stylesheet
    "https://codepen.io/chriddyp/pen/bWLwgP.css",
    "https://fonts.googleapis.com/css?family=Lobster|Raleway",
    "//maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
]
app = dash.Dash(name=app_name,
                server=server,
                url_base_pathname='/dashboard/',
                external_stylesheets=external_stylesheets)
server.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']  # On heroku the uri may change frequently.

db.init_app(server)
migrate = Migrate(server, db)

# 2. Add layout property.
from src.ucmwebapp.layout import serve_layout
app.layout = serve_layout

# 3. add callbacks
from src.ucmwebapp import callbacks

for js in external_js:
    app.scripts.append_script({"external_url": js})
for css in external_stylesheets:
    app.css.append_css({"external_url": css})
