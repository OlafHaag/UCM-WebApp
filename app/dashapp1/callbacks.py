import io
import base64
import datetime
from collections import namedtuple
from hashlib import md5
from contextlib import contextmanager, wraps

from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.exc import IntegrityError

import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table

import pandas as pd

from app.extensions import db
from app.models import Device, User, CTSession, CircleTask

####################
# Helper functions #
####################
@contextmanager
def ignored(*exceptions):
    """ Context in which raised exceptions are ignored.
    Note, that all statements after the one raising the exception within the context are passed.
    """
    try:
        yield
    except exceptions:
        pass


def ignoreKeyError(func):
    """ Decorator that on excepting a KeyError or AttributeError within the function returns None.
    Meant for use with DataFrames.
    """
    @wraps(func)
    def new_func(*args):
        res = None
        with ignored(KeyError, AttributeError):
            res = func(*args)
        return res
    return new_func


##############
# DB related #
##############
def get_table_type(filename):
    """ Accepted filenames:
    device-<device_id>.csv
    user.csv
    session-<time_iso>.csv
    trials-<time_iso>-Block_<n>.csv
    """
    basename, ext = filename.split('.')
    parts = basename.split('-')
    first = parts[0]
    if ext == 'csv' and first in ['device', 'user', 'session', 'trials']:
        return first
    else:
        return None
    

def get_one_or_create(model,
                      create_method='',
                      create_method_kwargs=None,
                      **kwargs):
    """ Get a row from the database or create one if not already present.
    Based on stackoverflow answer: https://stackoverflow.com/a/37419325
    
    This implementation tackles a problem that actually should not happen with our kind of data:
    Someone else is committing the same data.
    
    To clarify: The device should be unique, as shall the user.
    Session should also be unique by their hash value, and probably time, too.
    For trials this function must not be used. We always add new trials if the session/block is not pre-existing.
    
    :param model: model class.
    :type model: SQLAlchemy.Model
    :param create_method: Function to use to create instance of model.
    :type create_method: function
    :param create_method_kwargs:
    :type create_method_kwargs: dict
    :return: namedtuple with Instance of model and if it was created or not.
    """
    result = namedtuple("Model", ["instance", "created"])
    try:
        return result(model.query.filter_by(**kwargs).one(), False)
    except NoResultFound:
        kwargs.update(create_method_kwargs or {})
        try:
            with db.session.begin_nested():
                created = getattr(model, create_method, model)(**kwargs)
                db.session.add(created)
            return result(created, True)
        except IntegrityError:
            return result(model.query.filter_by(**kwargs).one(), False)
        
        
def get_models(df, table_type):
    # ToDo: rework this, does it make sense to return "models"?
    models = list()

    if table_type == 'device':
        row0 = df.loc[0]  # We expect it to contain only 1 device.
        models.append(new_device(row0))
    elif table_type == 'user':
        row0 = df.loc[0]  # We expect it to contain only 1 user.
        models.append(new_user(row0))
    elif table_type == 'session':
        if df.task == 'CircleTask':
            for row in df.itertuples(index=False):
                models.append(new_ct_session(row))
    elif table_type == 'trials':
        for row in df.itertuples(index=True):
            models.append(new_circletask_trial(row))
    return models


@ignoreKeyError
def get_device(data):
    """ Gets either device from database or returns new one.
    
    :param data: Data for instance.
    :type data: pandas.Series
    :return: Device instance.
    :rtype: Device
    """
    device = Device.query.get(data.id)
    if not device:
        device = new_device(data)
    return device
    
    
@ignoreKeyError
def new_device(data):
    """ Create a new instance of Device.

    :param data: Data for instance.
    :type data: pandas.Series
    :return: Device instance.
    :rtype: Device
    """
    device = Device(id=data.device,
                    screen_x=data.screen_x,
                    screen_y=data.screen_y,
                    dpi=data.dpi,
                    density=data.density,
                    aspect_ratio=data.aspect_ratio,
                    size_x=data.size_x,
                    size_y=data.size_y,
                    platform=data.platform)
    return device


@ignoreKeyError
def new_user(data):
    """ Create a new instance of User.

    :param data: Data for instance.
    :type data: pandas.Series
    :return: User instance.
    :rtype: User
    """
    user = User(id=data.id,
                device_id=data.device)
    return user


@ignoreKeyError
def new_ct_session(data):
    """ Create a new instance of CTSession (block).
    
    :param data: Data for instance.
    :type data: pandas.Series
    :return: CTSession instance.
    :rtype: CTSession
    """
    session = CTSession(block=data.block,
                        treatment=data.treatment,
                        time=data.time,
                        time_iso=data.time_iso,
                        hash=data.hash)
    return session


@ignoreKeyError
def new_circletask_trial(data):
    """ Create a new instance of CircleTask.
    
    :param data: Data for instance.
    :type data: pandas.Series
    :return: CircleTask instance.
    :rtype: CircleTask
    """
    trial = CircleTask(trial=data.Index,
                       df1=data.df1,
                       df2=data.df2)
    return trial


##############
# Without DB #
##############
def get_file_indices(filenames, table_type):
    """ Get indices of given table type from a list of file names.
    
    :param filenames: Received file names.
    :type filenames: list
    :param table_type: Which tables to look for.
    :type table_type: str
    :return: Indices of found files.
    :rtype: list
    """
    indices = [i for i, f in enumerate(filenames) if get_table_type(f) == table_type]
    return indices


def content_to_df(content):
    """ Decode uploaded file content and return Dataframe.
    
    :param content: base64 encoded content of a file.
    :type content: bytes
    :return: Converted Dataframe
    :rtype: pandas.DataFrame
    """
    decoded = base64.b64decode(content)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return None
    return df


def check_circletask_integrity(df, sent_hash, default=0.1):
    """ Evaluate if this data was tampered with and the controls actually touched.
    
    :param df: Dataframe to check.
    :type df: pandas.DataFrame
    :param sent_hash: The received hash to compare df to.
    :type sent_hash: str
    :param default: The default value for the slider. If it's still the same, the task wasn't done properly.
    :type default: float
    :return: Status of integrity.
    :rtype: bool
    """
    check = check_circletask_touched(df, default) and check_circletask_hash(df, sent_hash)
    return check


def check_circletask_hash(df, sent_hash):
    """ Check if hashes match.
    
    :param df: Dataframe to check.
    :type df: pandas.DataFrame
    :param sent_hash: The received hash to compare df to.
    :type sent_hash: str
    :return: Do hashes match?
    :rtype: bool
    """
    df_hash = md5(df.round(5).values.copy(order='C')).hexdigest()
    if df_hash == sent_hash:
        return True
    else:
        return False


def check_circletask_touched(df, default=0.1):
    """ Check if trials are all still on at default.
    
    :param df: Dataframe to check.
    :type df: pandas.DataFrame
    :param default: The default value for the slider. If it's still the same, the task wasn't done properly.
    :type default: float
    :return: If the sliders where used at all.
    :rtype: bool
    """
    touched = not (df == default).all()
    return touched
    
    
def parse_upload_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    
    decoded = base64.b64decode(content_string)
    # ToDo:
    #       if integrity not given, return error (data corrupted).
    #       Get/create device and user entries already exist in db.
    #       device.users.append(user_instance)
    #       Check if session is already present, if yes: abort & return error (duplicate).
    #       ctsession_inst.trials_CT.extend(trials)
    #       if not session.add_all(devices+users+sessions+trials)
    #       Get df from updated db.
    #       Show plot.
    error_div = html.Div(['There was an error processing this file.'])
    table_type = get_table_type(filename)
    if table_type:
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            print(e)
            return error_div
    else:
        return error_div
    
    #models = get_models(df, table_type)
    
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),
        
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),
        
        html.Hr(),  # horizontal line
        
        # For debugging, display the raw contents provided by the web browser
        #html.Div('Raw Content'),
        #html.Pre(contents[0:200] + '...', style={
        #    'whiteSpace': 'pre-wrap',
        #    'wordBreak': 'break-all'
        #})
    ])


################
# UI Callbacks #
################
def register_callbacks(dashapp):
    @dashapp.callback(Output('output-data-upload', 'children'),
                      [Input('upload-data', 'contents')],
                      [State('upload-data', 'filename'),
                       State('upload-data', 'last_modified')])
    def update_output(list_of_contents, list_of_names, list_of_dates):
        """ Handles files that are sent to the dash update component."""
        if list_of_contents is not None:
            # ToDo: return plots.
            children = [
                parse_upload_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]
            return children
