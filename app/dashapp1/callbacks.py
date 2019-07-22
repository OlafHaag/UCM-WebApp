import io
import base64
import datetime
from collections import namedtuple
from hashlib import md5
from contextlib import suppress, wraps

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
def none_on_error(func):
    """ Decorator that on excepting a KeyError or AttributeError within the function returns None.
    Meant for use with DataFrames and Series.
    """
    @wraps(func)
    def new_func(*args):
        res = None
        with suppress(KeyError, AttributeError):
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
                      create_func=None,
                      defaults=None,
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
    :param create_func: function to use to create new instance of model. If None, model constructor is used.
    :type create_func: function
    :param defaults: If no result was found, update kwargs with these defaults.
    :type defaults: dict
    :return: namedtuple with Instance of model and if it was created or not.
    :rtype: tuple
    """
    result = namedtuple("Model", ["instance", "created"])
    try:
        return result(model.query.filter_by(**kwargs).one(), False)
    except NoResultFound:
        kwargs.update(defaults or {})
        try:
            with db.session.begin_nested():
                if create_func:
                    created = create_func(**kwargs)
                else:
                    created = model(**kwargs)
                if created:
                    db.session.add(created)
            return result(created, True)
        except IntegrityError:
            return result(model.query.filter_by(**kwargs).one(), False)
        

@none_on_error
def new_device(**kwargs):
    """ Create a new instance of Device.

    :return: Device instance.
    :rtype: Device
    """
    device = Device(**kwargs)
    return device


@none_on_error
def new_user(**kwargs):
    """ Create a new instance of User.

    :return: User instance.
    :rtype: User
    """
    user = User(**kwargs)
    return user


@none_on_error
def new_ct_session(**kwargs):
    """ Create a new instance of CTSession (block).
    
    :return: CTSession instance.
    :rtype: CTSession
    """
    session = CTSession(**kwargs)
    return session


@none_on_error
def new_circletask_trial(**kwargs):
    """ Create a new instance of CircleTask.
    
    :return: CircleTask instance.
    :rtype: CircleTask
    """
    trial = CircleTask(trial=kwargs['Index'],
                       df1=kwargs['df1'],
                       df2=kwargs['df2'])
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
    error_div = html.Div(["ERROR: There was an error during file processing."])
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


def get_idx_or_error(filenames, table_type):
    """ Supposed to be used against table types 'device', 'user' and 'session' which should yield a single file.
    
    :param filenames:
    :type filenames: list
    :param table_type:
    :type table_type: str
    :return:
    :rtype: int| html.Div
    """
    try:
        file_idx = get_file_indices(filenames, table_type)[0]
    except IndexError:
        return html.Div([f"ERROR: {table_type.title()} file missing."])
    return file_idx


def decode_contents(list_of_contents):
    """ Decode list of base64 encoded uploaded data and convert it to decoded list of data.
    
    :param list_of_contents: List of encoded content from dash upload component.
    :type list_of_contents: list
    :return: List of decoded contents (str)
    :rtype: list
    """
    decoded = list()
    for contents in list_of_contents:
        content_type, content_string = contents.split(',')
        decoded.append(base64.b64decode(content_string).decode('utf-8'))
    return decoded

    
def parse_uploaded_files(list_of_contents, list_of_filenames):
    
    # If files are missing return error message.
    device_file_idx = get_idx_or_error(list_of_filenames, 'device')
    if not isinstance(device_file_idx, int):
        return [device_file_idx]
    user_file_idx = get_idx_or_error(list_of_filenames, 'user')
    if not isinstance(user_file_idx, int):
        return [user_file_idx]
    session_file_idx = get_idx_or_error(list_of_filenames, 'session')
    if not isinstance(session_file_idx, int):
        return [session_file_idx]
    trials_file_idxs = get_file_indices(list_of_filenames, 'trials')
    if not trials_file_idxs:
        return [html.Div(["ERROR: Trial files missing."])]
    
    # Generic error message.
    error_div = html.Div(["ERROR: There was an error during file processing."])
    
    try:  # I know this is a big block, but I didn't want to wrap each statement in a try statement.
        # Decode the content.
        decoded_list = decode_contents(list_of_contents)
        
        # Extract data from content for the database models.
        # For device and user we expect them to contain only 1 entry.
        device_data = pd.read_csv(io.StringIO(decoded_list[device_file_idx])).iloc[0]
        user_data = pd.read_csv(io.StringIO(decoded_list[user_file_idx])).iloc[0]
        session_df = pd.read_csv(io.StringIO(decoded_list[session_file_idx]))
        trials_dfs = [pd.read_csv(io.StringIO(decoded_list[idx])) for idx in trials_file_idxs]
    except Exception as e:
        print(e)
        return error_div

    # ToDo: Check if any of these are None.
    device = get_one_or_create(Device, create_func=new_device, **device_data.to_dict()).instance
    user = get_one_or_create(User, create_func=new_user, **user_data.to_dict()).instance
    sessions = [get_one_or_create(CTSession, create_func=new_ct_session, **data._asdict()).instance
                for data in session_df.itertuples(index=False)]  # itertuples doesn't return Series.
    # ToDo: Check if sessions were already uploaded.


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
