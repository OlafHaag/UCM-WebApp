"""App config."""
import os
from pathlib import Path

from dotenv import load_dotenv

from exceptions import ImproperlyConfigured


DOTENV_PATH = Path(__file__).parent / '.env'
load_dotenv(DOTENV_PATH)


class BaseConfig:
    """Global configuration variables."""

    # General Config
    try:
        SECRET_KEY = os.environ['SECRET_KEY']
    except KeyError:
        raise ImproperlyConfigured("SECRET_KEY not set in .env:")
    
    try:
        FLASK_APP = os.environ.get('FLASK_APP')
    except KeyError:
        raise ImproperlyConfigured("FLASK_APP not set in .env:")
    
    try:
        FLASK_ENV = os.environ.get('FLASK_ENV')
    except KeyError:
        raise ImproperlyConfigured("FLAK_ENV not set in .env:")

    # Database
    try:
        SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')
    except KeyError:
        raise ImproperlyConfigured("DATABASE_URL not set in .env:")

    # Silence deprecation warning.
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Assets
    LESS_BIN = os.environ.get('LESS_BIN')
    ASSETS_DEBUG = os.environ.get('ASSETS_DEBUG')
    LESS_RUN_IN_DEBUG = os.environ.get('LESS_RUN_IN_DEBUG')
    
    # Static Assets
    STATIC_FOLDER = os.environ.get('STATIC_FOLDER')
    TEMPLATES_FOLDER = os.environ.get('TEMPLATES_FOLDER')
    COMPRESSOR_DEBUG = os.environ.get('COMPRESSOR_DEBUG')
