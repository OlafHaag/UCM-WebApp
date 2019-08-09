from flask import Blueprint, render_template
from flask import current_app as app


# Blueprint Configuration
privacy_bp = Blueprint('privacy_bp', __name__,
                       template_folder='../templates',
                       static_folder='../static')


@privacy_bp.route('/privacypolicy', methods=['GET'])
def privacy():
    """ Privacy policy page route."""
    return render_template('index.html',
                           title='Privacy Policy',
                           template='privacy-template',
                           body="All your base are belong to us.")
