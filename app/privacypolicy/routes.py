import os
from pathlib import Path

from flask import Blueprint, render_template
from flask import current_app as app

author = os.environ.get('AUTHOR')
street = os.environ.get('STREET')
postal = os.environ.get('POSTAL')
city = os.environ.get('CITY')
email = os.environ.get('EMAIL')

templates_path = Path(__file__).parent / "templates"
information_html = (templates_path / "information.html").read_text(encoding='utf-8').format(name=author, email=email)

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
                           body=information_html)
