"""Routes for core Flask app."""
from flask import Blueprint, render_template
from flask_assets import Environment, Bundle
from flask import current_app as app

# Flask-Assets Configuration
assets = Environment(app)
Environment.auto_build = True
Environment.debug = False
less_bundle = Bundle('less/*.less',
                     filters='less,cssmin',
                     output='dist/css/styles.css',
                     extra={'rel': 'stylesheet/less'})
js_bundle = Bundle('js/*.js',
                   filters='jsmin',
                   output='dist/js/main.js')
assets.register('less_all', less_bundle)
# ToDo: Merge minified files together with uncompressed ones (jquery.min.js)?
#       Docs suggest assets.register('js_all', 'jquery.min.js', js_bundle), but doesn't work.
assets.register('js_all', js_bundle)

# ToDo: bundle dashapp1.less separately? So it's not loaded when not needed, i.e. on other pages.
#       Maybe use assets.append_path?

if app.config['FLASK_ENV'] == 'development':
    # Convert static/less/*.less files into static/dist/styles.css
    less_bundle.build(force=True)
    js_bundle.build()
    
# Blueprint Configuration
main_bp = Blueprint('main_bp', __name__,
                    template_folder='templates',
                    static_folder='static')


@main_bp.route('/')
def home():
    """Landing page."""
    # ToDo: Landing page text.
    return render_template('index.html',
                           title='Uncontrolled Manifold Analysis.',
                           template='home-template',
                           body="This website serves as a visualization tool for the analysis of my UCM research.")
