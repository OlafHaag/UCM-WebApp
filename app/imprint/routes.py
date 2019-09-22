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
contact_html = (templates_path / "contact.html").read_text(encoding='utf-8').format(name=author,
                                                                                    email=email,
                                                                                    street=street,
                                                                                    postal=postal,
                                                                                    city=city)
info_html = (templates_path / "information.html").read_text()

# Blueprint Configuration
imprint_bp = Blueprint('imprint_bp', __name__,
                       template_folder='../templates',
                       static_folder='../static')


@imprint_bp.route('/imprint', methods=['GET'])
def privacy():
    """ Privacy policy page route."""
    return render_template('index.html',
                           title='ucmwebapp|Imprint',
                           template='imprint-template',
                           body=contact_html + info_html)
