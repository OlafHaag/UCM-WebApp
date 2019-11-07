import pytest
import dash_html_components as html
from app import create_app
from exceptions import ImproperlyConfigured


def f():
    raise ImproperlyConfigured('TODO')


def test_exception_is_raised():
    with pytest.raises(ImproperlyConfigured):
        f()


def test_debug_is_disabled():
    app = create_app()
    assert app.server.debug is False


def test_layout_is_a_function_that_returns_a_div_element():
    app = create_app()
    assert isinstance(app.layout(), html.Div)
