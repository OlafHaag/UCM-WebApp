"""Initialize app."""
import dash
from flask import Flask
from flask.helpers import get_root_path

from config import BaseConfig


def create_app(**config_overrides):
    server = Flask(__name__, instance_relative_config=False)
    server.config.from_object(BaseConfig)
    server.config.update(config_overrides)

    with server.app_context():
        # Before loading layouts we need to connect to the database.
        register_extensions(server)
    
        # Add the first dash application to the flask server.
        from app.circletask.layout import serve_layout as layout1
        from app.circletask.layout import html_layout as dash_ct_index
        from app.circletask.layout import app_route as dash_ct_route
        from app.circletask.callbacks import register_callbacks as register_callbacks1
        from app.circletask.external import external_stylesheets as stylesheets1
        from app.circletask.external import external_scripts as scripts1
        register_dashapp(server,
                         'CircleTaskAnalysis',
                         dash_ct_route,
                         layout1,
                         register_callbacks1,
                         dash_ct_index,
                         stylesheets1,
                         scripts1)
    
        # Add here any other dash apps.
        
        # Add other functionality to the flask app.
        register_blueprints(server)
    
        return server


def register_dashapp(app,
                     title,
                     base_pathname,
                     layout,
                     register_callbacks_func,
                     index_layout=None,
                     stylesheets=None,
                     scripts=None):
    # Meta tags for viewport responsiveness
    meta_viewport = {"name": "viewport", "content": "width=device-width, initial-scale=1, shrink-to-fit=no"}
    
    if not stylesheets:
        stylesheets = list()
        
    if not scripts:
        scripts = list()
    
    dash_app = dash.Dash(__name__,
                         server=app,
                         url_base_pathname=f'/{base_pathname}/',
                         external_stylesheets=stylesheets,
                         external_scripts=scripts,
                         assets_folder=get_root_path(__name__) + f'/{base_pathname}/assets/',
                         meta_tags=[meta_viewport])

    with app.app_context():
        dash_app.title = title
        # Override the underlying HTML template
        if index_layout:
            dash_app.index_string = index_layout
        dash_app.layout = layout
        register_callbacks_func(dash_app)


def register_extensions(server):
    from app.extensions import db
    # We need to import the models for migration to detect them.
    from app.models import Device, User, CTSession, CircleTask
    from app.extensions import migrate

    db.init_app(server)
    migrate.init_app(server, db)


def register_blueprints(server):
    from app.routes import main_bp
    from app.privacypolicy.routes import privacy_bp
    from app.imprint.routes import imprint_bp

    server.register_blueprint(main_bp)
    server.register_blueprint(privacy_bp)
    server.register_blueprint(imprint_bp)
