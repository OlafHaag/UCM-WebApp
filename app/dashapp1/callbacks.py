import io
import base64
from datetime import datetime
from collections import namedtuple
from hashlib import md5
from contextlib import suppress, wraps

from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.exc import IntegrityError

import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table

import pandas as pd
import numpy as np
from psycopg2.extensions import register_adapter, AsIs

from app.extensions import db
from app.models import Device, User, CTSession, CircleTask
from .exceptions import UploadError, ModelCreationError


# Numpy data types compatibility with postgresql database.
register_adapter(np.int64, AsIs)
register_adapter(np.float64, AsIs)

time_fmt = '%Y_%m_%d_%H_%M_%S'

# ToDo: use df.to_sql() for trials?
#       Get df from updated db.
#       Show plot.
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


def raise_on_error(func):
    """ Decorator that raises ModelCreationError on KeyError, AttributeError or TypeError.
    Meant for use with functions for creating db.Model instances.
    """
    @wraps(func)
    def new_func(*args, **kwargs):
        res = None
        try:
            res = func(*args, **kwargs)
        except (KeyError, AttributeError, TypeError):
            raise ModelCreationError("ERROR: Insufficient data.")
        return res
    return new_func


##############
# DB related #
##############
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
                    new_instance = create_func(**kwargs)  # Throws ModelCreationError if kwargs not sufficient.
                else:
                    new_instance = model(**kwargs)  # Doesn't check if there's sufficient data provided.
                if new_instance:
                    db.session.add(new_instance)
            return result(new_instance, True)
        except IntegrityError:
            try:
                return result(model.query.filter_by(**kwargs).one(), False)
            except NoResultFound:
                raise ModelCreationError("ERROR: Integrity compromised.\nFailed to get or create model.")
    

@raise_on_error
def new_device(**kwargs):
    """ Create a new instance of Device.

    :return: Device instance.
    :rtype: Device
    """
    device = Device(id=kwargs['id'],  # FixMe: setting primary key doesn't seem to work?
                    screen_x=kwargs['screen_x'],
                    screen_y=kwargs['screen_y'],
                    dpi=kwargs['dpi'],
                    density=kwargs['density'],
                    aspect_ratio=kwargs['aspect_ratio'],
                    size_x=kwargs['size_x'],
                    size_y=kwargs['size_y'],
                    platform=kwargs['platform'])
    return device
        

@raise_on_error
def new_user(**kwargs):
    """ Create a new instance of User.

    :return: User instance.
    :rtype: User
    """
    user = User(id=kwargs['id'],  # FixMe: setting primary key doesn't seem to work?
                device_id=kwargs['device'])  # FixMe: This won't work. Needs ORM object.
    return user


@raise_on_error
def new_ct_session(**kwargs):
    """ Create a new instance of CTSession (block).
    
    :return: CTSession instance.
    :rtype: CTSession
    """
    session = CTSession(block=kwargs['block'],
                        treatment=kwargs['treatment'],
                        time=kwargs['time'],
                        time_iso=kwargs['time_iso'],
                        hash=kwargs['hash'])
    return session


@raise_on_error
def new_circletask_trial(**kwargs):
    """ Create a new instance of CircleTask.
    
    :return: CircleTask instance.
    :rtype: CircleTask
    """
    trial = CircleTask(user_id=kwargs['user_id'],    # FixMe: This won't work. Needs ORM object.
                       trial=kwargs['Index'],
                       df1=kwargs['df1'],
                       df2=kwargs['df2'])
    return trial


def add_to_db(device_kwargs, user_kwargs, session_kwargs, trials_kwargs):
    # FixMe: Optimize with df.to_sql(name=sql_table_name, con=db.engine, if_exists='fail', index=False, method='multi').
    #   ValueError when the table already exists and if_exists is ‘fail’.
    # Create model instances.
    try:
        device = get_one_or_create(Device, create_func=new_device, **device_kwargs).instance
        user, is_new_user = get_one_or_create(User, create_func=new_user, **user_kwargs)
    except ModelCreationError:
        raise
    
    if is_new_user and device.id == user.device_id:
        device.users.append(user)
    
    # Retrieve 2 tuples for sessions. The first contains session instances,
    # the second contains information about whether they weren't already in the db.
    sessions, created = list(zip(*[get_one_or_create(CTSession, create_func=new_ct_session, **kw)
                                   for kw in session_kwargs]))
    
    # Check if sessions were already uploaded.
    if not all(created):
        raise ModelCreationError("ERROR: Session was already uploaded.")
    else:
        # Only if sessions are new, add trials.
        trials = list()
        for kw in trials_kwargs:
            try:
                session_idx = kw.pop('session_idx')
            except KeyError:
                raise ModelCreationError("ERROR: Failed to relate trial to session.")
            try:
                trial, created = get_one_or_create(CircleTask, create_func=new_circletask_trial, **kw)
            except ModelCreationError:
                raise
            # Add trial to corresponding session.
            sessions[session_idx].trials_CT.append(trial)
    
    db.session.commit()
            

############################
# Map files to table types #
############################
def get_table_type(filename):
    """ Accepted filenames:
    device-<device_id>.csv,
    user.csv,
    session-<time_iso>.csv,
    trials-<time_iso>-Block_<n>.csv
    
    :param filename: name of uploaded file.
    :type filename: str
    :return: Name of table type.
    :rtype: str|None
    """
    basename, ext = filename.split('.')
    parts = basename.split('-')
    first = parts[0]
    if ext == 'csv' and first in ['device', 'user', 'session', 'trials']:
        return first
    else:
        return None
    
    
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


def get_table_idx(filenames, table_type):
    """ Supposed to be used against table types 'device', 'user' and 'session' which should yield a single file.
    
    :param filenames:
    :type filenames: list
    :param table_type:
    :type table_type: str
    :return: Index of single table.
    :rtype: int
    """
    try:
        file_idx = get_file_indices(filenames, table_type)[0]
    except IndexError:
        raise UploadError(f"ERROR: {table_type.title()} file is missing.")
    return file_idx


def get_idx_dict(list_of_filenames):
    """ Take list of file names and map table types to the list indices.
    
    :param list_of_filenames: List of received file names.
    :return: Dictionary with indices of uploaded files in list: device, user, session, trials
    :rtype: dict
    """
    indices = dict()
    # If files are missing get_table_idx raises UploadError.
    indices['device'] = get_table_idx(list_of_filenames, 'device')
    indices['user'] = get_table_idx(list_of_filenames, 'user')
    indices['session'] = get_table_idx(list_of_filenames, 'session')
    indices['trials'] = get_file_indices(list_of_filenames, 'trials')
    if not indices['trials']:
        raise UploadError("ERROR: Trial files are missing.")
    return indices


#########################
# Extract file contents #
#########################
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
        return None  # ToDo raise Error instead.
    return df


####################
# Integrity Checks #
####################
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


def get_trials_meta(filename):
    """ Take a filename and extract time_iso and block information from it.
    
    :param filename: file name of trials table of form trials-[time_iso]-Block_[n].csv
    :type filename: str
    :return: time_iso and block as namedtuple
    :rtype: tuple
    """
    basename, ext = filename.split('.')
    parts = basename.split('-')
    try:
        time_iso = datetime.strptime(parts[1], time_fmt)  # Convert string to datetime.
        block = int(parts[2].split('_')[1])
    except (IndexError, ValueError):
        raise UploadError("ERROR: Trial table file name has to be of form: trials-<time_iso>-Block_<n>.csv")
    meta = namedtuple('trialMeta', ['time_iso', 'block'])
    return meta(time_iso, block)
    
    
################
# Parsing main #
################
def parse_uploaded_files(list_of_filenames, list_of_contents):
    """ Reads files and returns dictionary with keyword arguments for each table type.
    These are:
    
        'device': dict
        
        'user': dict
        
        'session': list
        
        'trials': generator
    
    :param list_of_filenames: list of received file names.
    :type list_of_filenames: list
    :param list_of_contents: List of encoded contents.
    :type list_of_contents: list
    :return: Dictionary with keyword arguments for database models.
    :rtype: dict
    """
    
    # If there was an error in mapping the table types to files, return that error.
    try:
        table_idx = get_idx_dict(list_of_filenames)
    except UploadError:
        raise
    
    # Decode the content.
    try:
        decoded_list = decode_contents(list_of_contents)
    except Exception:
        raise UploadError("ERROR: Failed to decode file contents.")
    
    # Extract data from content for the database models.
    kw = dict()  # Keyword arguments for each table.
    try:
        # For device and user we expect them to contain only 1 entry.
        kw['device'] = pd.read_csv(io.StringIO(decoded_list[table_idx['device']])).iloc[0].to_dict()  # df->Series->dict
    except Exception:
        raise UploadError("ERROR: Failed to read file contents for device.")
    try:
        kw['user'] = pd.read_csv(io.StringIO(decoded_list[table_idx['user']])).iloc[0].to_dict()  # df->Series->dict
    except Exception:
        raise UploadError("ERROR: Failed to read file contents for user.")
    try:
        session_df = pd.read_csv(io.StringIO(decoded_list[table_idx['session']]))
        # Convert time_iso string to datetime.
        session_df['time_iso'] = session_df['time_iso'].apply(lambda t: datetime.strptime(t, time_fmt))
        session_df['user_id'] = kw['user']['id']
    except Exception:
        raise UploadError("ERROR: Failed to read file contents for session.")
    # We expect only a few blocks for a session, so put it all in one list.
    kw['session'] = [data._asdict() for data in session_df.itertuples(index=False)]  # Convert namedtuple to dict.
    
    trials_dfs = list()
    for idx in table_idx['trials']:
        # Get information from filename.
        try:
            trials_meta = get_trials_meta(list_of_filenames[idx])
        except UploadError:
            raise
        # Get data from content.
        try:
            df = pd.read_csv(io.StringIO(decoded_list[idx]))
        except Exception:
            raise UploadError("ERROR: Failed to read file contents for trials.")
        # Add index of session for later relationship assignment.
        mask = (session_df['time_iso'] == trials_meta.time_iso) & (session_df['block'] == trials_meta.block)
        try:
            df['session_idx'] = session_df[mask].index[0]
        except IndexError:
            raise UploadError("ERROR: Mismatch between session data and trials.")
        trials_dfs.append(df)
        
    # Concatenate the different trials DataFrames. Rows are augmented by block & time_iso for differentiation later on.
    df = pd.concat(trials_dfs)
    df['user_id'] = kw['user']['id']
    # We may get a lot of trials, put them in a generator to not hold a large list of dictionaries in memory.
    kw['trials'] = (row._asdict() for row in df.itertuples(index=True))
    
    return kw
    
    
def process_upload(filenames, contents):
    """
    
    :param filenames: list of received file names.
    :type filenames: list
    :param contents: List of encoded contents.
    :type contents: list
    """
    # Get keyword arguments for creation of each model.
    try:
        kw = parse_uploaded_files(filenames, contents)
    except UploadError:
        raise
    
    try:
        add_to_db(kw['device'], kw['user'], kw['session'], kw['trials'])
    except ModelCreationError:
        raise
    
    
def get_db_data():
    pass


def plot_data():
    pass

    
# ToDo: replace, make obsolete!
def parse_upload_contents(contents, filename, date):
    content_type, content_string = contents.split(',')
    
    decoded = base64.b64decode(content_string)
    
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return html.Div(["ERROR: There was an error during file processing."])
    
    return html.Div([
        html.H5(filename),
        html.H6(datetime.fromtimestamp(date)),
        
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
    
            try:
                #process_upload(list_of_names, list_of_contents)
                children = [
                    parse_upload_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]
            except (UploadError, ModelCreationError) as e:
                return [html.Div([str(e)])]  # Display the error message.
    
            # ToDo: return plots.
    
            return children
