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
from .analysis import get_data

# Numpy data types compatibility with postgresql database.
register_adapter(np.int64, AsIs)
register_adapter(np.float64, AsIs)

time_fmt = '%Y_%m_%d_%H_%M_%S'


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
    device = Device(id=kwargs['id'],
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
    user = User(id=kwargs['id'],
                device_id=kwargs['device_id'])
    return user


@raise_on_error
def new_ct_session(**kwargs):
    """ Create a new instance of CTSession (block).
    
    :return: CTSession instance.
    :rtype: CTSession
    """
    # ToDo: session count
    session = CTSession(user_id=kwargs['user_id'],
                        block=kwargs['block'],
                        treatment=kwargs['treatment'],
                        warm_up=kwargs['warm_up'],
                        trial_duration=kwargs['trial_duration'],
                        cool_down=kwargs['cool_down'],
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
    trial = CircleTask(user_id=kwargs['user_id'],
                       session=kwargs['session'],
                       trial=kwargs['trial'],
                       df1=kwargs['df1'],
                       df2=kwargs['df2'])
    return trial


def add_to_db(device_kwargs, user_kwargs, session_kwargs, trials_kwargs):
    # Create model instances.
    try:
        device = get_one_or_create(Device, create_func=new_device, **device_kwargs).instance
        user, is_new_user = get_one_or_create(User, create_func=new_user, **user_kwargs)
    except ModelCreationError:
        raise
    
    # Add sessions to db.
    task_err_msg = "ERROR: Failed to identify session as 'Circle Task'."
    sessions = list()
    for kw in session_kwargs:
        # This dashboard only accepts Circle Task (for now).
        try:
            task = kw.pop('task')
        except KeyError:
            raise ModelCreationError(task_err_msg)
        if task == 'Circle Task':
            model = CTSession
        else:
            raise ModelCreationError(task_err_msg)
        try:
            session, created = get_one_or_create(model, create_func=new_ct_session, **kw)
        except ModelCreationError:
            raise
        # Check if session was already uploaded.
        if not created:
            raise ModelCreationError("ERROR: Session(s) already uploaded.")
        sessions.append(session)
    
    # Add trials.
    # ToDo: Use df.to_sql(name='circle_tasks', con=db.engine, if_exists='append', index=False, method='multi')?
    trials = list()
    for kw in trials_kwargs:
        try:
            session_idx = kw.pop('session_idx')
        except KeyError:
            raise ModelCreationError("ERROR: Failed to relate trial to session.")
        # Add session relationship.
        kw['session'] = sessions[session_idx].id
        try:
            trial, created = get_one_or_create(CircleTask, create_func=new_circletask_trial, **kw)
        except ModelCreationError:
            raise
    
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
    df_hash = md5(df[['df1', 'df2']].round(5).values.copy(order='C')).hexdigest()
    if df_hash != sent_hash:
        raise UploadError("ERROR: Data corrupted.")
    else:
        return True
        

def check_circletask_touched(df, default=10.0):
    """ Check if trials are all still on at default.

    :param df: Dataframe to check.
    :type df: pandas.DataFrame
    :param default: The default value for the slider. If it's still the same, the task wasn't done properly.
    :type default: float
    :return: If the sliders where used at all.
    :rtype: bool
    """
    touched = not (df[['df1', 'df2']].values == default).all()
    if not touched:
        raise UploadError("ERROR: Task not properly executed.")
    return True


def check_circletask_integrity(df, sent_hash, default=10.0):
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
    try:
        check = check_circletask_touched(df, default) and check_circletask_hash(df, sent_hash)
    except UploadError:
        raise
    return check


##################
# Parsing upload #
##################
def get_device_properties(csv_file):
    """ Get device properties as dictionary from a CSV file.
    
    :param csv_file: 1 row table with device properties.
    :type csv_file: str|io.StringIO
    :return: Properties of the device.
    :rtype: dict
    """
    try:
        # For device and user we expect them to contain only 1 entry.
        props = pd.read_csv(csv_file).iloc[0].to_dict()  # df->Series->dict
    except Exception:
        raise UploadError("ERROR: Failed to read file contents for device.")
    return props
    
    
def get_user_properties(csv_file):
    """ Get user data as dictionary from a CSV file.

    :param csv_file: 1 row table with user data.
    :type csv_file: str|io.StringIO
    :return: Properties of the user.
    :rtype: dict
    """
    try:
        # For device and user we expect them to contain only 1 entry.
        props = pd.read_csv(csv_file).iloc[0].to_dict()  # df->Series->dict
    except Exception:
        raise UploadError("ERROR: Failed to read file contents for user.")
    return props


def get_session_df(csv_file, user_id):
    """ Return a DataFrame from CSV file or buffer and add user_id column.
    
    :param csv_file: Table with session data.
    :type csv_file: str|io.StringIO
    :param user_id: ID of the user who performed the session.
    :type user_id: str
    :return: Properties of the blocks.
    :rtype: pandas.DataFrame
    """
    try:
        # Read data and keep empty string in treatment as empty string, not NaN.
        session_df = pd.read_csv(csv_file, keep_default_na=False)
        # Convert time_iso string to datetime.
        session_df['time_iso'] = session_df['time_iso'].apply(lambda t: datetime.strptime(t, time_fmt))
        session_df['user_id'] = user_id
    except Exception:
        raise UploadError("ERROR: Failed to read file contents for session.")
    return session_df

    
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
    

def get_trials_properties(filenames, contents, times, blocks, hashes, user_id):
    """
    
    :param filenames: All uploaded filenames.
    :type filenames: list
    :param contents: Decoded file contents.
    :type contents: list
    :param times: 'time_iso' column of session.
    :type times: pandas.Series
    :param blocks: 'block' column of session.
    :type blocks: pandas.Series
    :param hashes: 'hash' column of session.
    :type hashes: pandas.Series
    :param user_id: ID of user who provided the data of trials.
    :type user_id: str
    :return: Properties of all the trials.
    :rtype: dict
    """
    file_indices = get_file_indices(filenames, 'trials')
    if not file_indices:
        raise UploadError("ERROR: Trial files are missing.")
    
    trials_dfs = list()
    for idx in file_indices:
        # Get information from filename.
        try:
            trials_meta = get_trials_meta(filenames[idx])
        except UploadError:
            raise
        # Get data from content.
        try:
            df = pd.read_csv(io.StringIO(contents[idx]))
        except Exception:
            raise UploadError("ERROR: Failed to read file contents for trials.")
        # Add index of session for later relationship assignment.
        mask = (times == trials_meta.time_iso) & (blocks == trials_meta.block)
        try:
            session_idx = mask[mask].index[0]
            df['session_idx'] = session_idx
        except IndexError:
            raise UploadError("ERROR: Mismatch between session data and trials.")
        # Check data integrity.
        sent_hash = hashes.iloc[session_idx]
        try:
            check_passed = check_circletask_integrity(df, sent_hash)
        except UploadError:
            raise
        trials_dfs.append(df)
    
    # Concatenate the different trials DataFrames. Rows are augmented by block & time_iso for differentiation later on.
    df = pd.concat(trials_dfs)
    df['user_id'] = user_id
    df['trial'] = df.index
    # We may get a lot of trials, put them in a generator to not hold a large list of dictionaries in memory.
    props = (row._asdict() for row in df.itertuples(index=False))
    return props


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
        # If files are missing get_table_idx raises UploadError.
        device_idx = get_table_idx(list_of_filenames, 'device')
        user_idx = get_table_idx(list_of_filenames, 'user')
        session_idx = get_table_idx(list_of_filenames, 'session')
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
        kw['device'] = get_device_properties(io.StringIO(decoded_list[device_idx]))
        kw['user'] = get_user_properties(io.StringIO(decoded_list[user_idx]))
        session_df = get_session_df(io.StringIO(decoded_list[session_idx]), kw['user']['id'])
        kw['session'] = [data._asdict() for data in session_df.itertuples(index=False)]  # Convert namedtuple to dict.
        kw['trials'] = get_trials_properties(list_of_filenames,
                                             decoded_list,
                                             session_df['time_iso'],
                                             session_df['block'],
                                             session_df['hash'],
                                             kw['user']['id'])
    except UploadError:
        raise
    
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
 
 
#################
# Visualization #
#################
def visualize_data(df):
    viz = html.Div([
        html.H5("Trials"),
        html.H6("Across participants and blocks."),
        
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),
        
        html.Hr(),  # horizontal line
    ])
    return viz
    

def plot_data(df):
    pass

    
def process_data():
    df = get_data()
    viz = visualize_data(df)
    return viz


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
                # Insert data into database.
                process_upload(list_of_names, list_of_contents)
                pass
            except (UploadError, ModelCreationError) as e:
                return [html.Div(str(e))]  # Display the error message.
            # FixMe: Figure out how to replace output-data-db table. If set as Output, it's not rendered initially.
            # ToDo: return plots.
            children = [html.Div("Upload successful.")]
            return children
