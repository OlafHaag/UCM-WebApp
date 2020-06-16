# Built-in imports
import io
import base64
from datetime import datetime
from collections import namedtuple
from hashlib import md5
from contextlib import suppress, wraps
# Third-party module imports.
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.exc import IntegrityError

import dash
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import pandas as pd
import numpy as np
from psycopg2.extensions import register_adapter, AsIs
# Own module imports.
from app.extensions import db
from app.models import Device, User, CircleTaskBlock, CircleTaskTrial
from .exceptions import UploadError, ModelCreationError
from . import analysis
from . import layout

# Numpy data types compatibility with postgresql database.
register_adapter(np.int64, AsIs)
register_adapter(np.float64, AsIs)
# Describe time format that we get, e.g. in file names, so we can convert it to datetime.
time_fmt = '%Y_%m_%d_%H_%M_%S'
# Set pandas plotting backend to ploty. Requires plotly >= 4.8.0.
pd.options.plotting.backend = 'plotly'


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
    Block should also be unique by their hash value, and probably time, too.
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
def new_circletask_block(**kwargs):
    """ Create a new instance of CircleTaskBlock.
    
    :return: CircleTaskBlock instance.
    :rtype: CircleTaskBlock
    """
    # Get all unique sessions by user not including this uid and count + 1.
    nth_sessions = CircleTaskBlock.query.filter(CircleTaskBlock.user_id == kwargs['user_id'],
                                                CircleTaskBlock.session_uid != kwargs['session_uid']).distinct(
                                                                                            'session_uid').count() + 1
    block = CircleTaskBlock(user_id=kwargs['user_id'],
                            session_uid=kwargs['session_uid'],
                            nth_session=nth_sessions,
                            nth_block=kwargs['nth_block'],
                            treatment=kwargs['treatment'],
                            warm_up=kwargs['warm_up'],
                            trial_duration=kwargs['trial_duration'],
                            cool_down=kwargs['cool_down'],
                            time=kwargs['time'],
                            time_iso=kwargs['time_iso'],
                            hash=kwargs['hash'])
    return block


@raise_on_error
def new_circletask_trial(**kwargs):
    """ Create a new instance of CircleTask.
    
    :return: CircleTask instance.
    :rtype: CircleTaskTrial
    """
    trial = CircleTaskTrial(user_id=kwargs['user_id'],
                            block_id=kwargs['block_id'],
                            trial=kwargs['trial'],
                            df1=kwargs['df1'],
                            df2=kwargs['df2'],
                            df1_grab=kwargs['df1_grab'],
                            df1_release=kwargs['df1_release'],
                            df2_grab=kwargs['df2_grab'],
                            df2_release=kwargs['df2_release'],
                            )
    return trial


def add_to_db(device_kwargs, user_kwargs, blocks_kwargs, trials_kwargs):
    """ Takes data, checks for existing records and adds them to database if they're new. Raises UploadError on error.
    
    :param device_kwargs: Data about device.
    :type device_kwargs: dict
    :param user_kwargs: Data about user.
    :type user_kwargs: dict
    :param blocks_kwargs: Data from blocks.
    :type blocks_kwargs: list[dict]
    :param trials_kwargs: Data from trials.
    :type trials_kwargs: list[OrderedDict]
    
    :rtype: None
    """
    # Create model instances.
    try:
        device = get_one_or_create(Device, create_func=new_device, **device_kwargs).instance
        user, is_new_user = get_one_or_create(User, create_func=new_user, **user_kwargs)
    except ModelCreationError:
        raise
    
    # Add blocks to db.
    task_err_msg = "ERROR: Failed to identify session as 'Circle Task'."
    blocks = list()
    for kw in blocks_kwargs:
        # This dashboard only accepts Circle Task (for now).
        try:
            task = kw.pop('task')
        except KeyError:
            raise ModelCreationError(task_err_msg)
        if task == 'Circle Task':
            model = CircleTaskBlock
        # Here'd be the place to add other tasks through elif.
        else:
            raise ModelCreationError(task_err_msg)
        try:
            block, created = get_one_or_create(model, create_func=new_circletask_block, **kw)
        except ModelCreationError:
            raise
        # Check if session was already uploaded.
        if not created:
            raise ModelCreationError("ERROR: Session(s) already uploaded.")
        blocks.append(block)
    
    # Add trials.
    # ToDo: Use df.to_sql(name='circle_tasks', con=db.engine, if_exists='append', index=False, method='multi')?
    trials = list()
    for kw in trials_kwargs:
        try:
            block_idx = kw.pop('block_idx')
        except KeyError:
            raise ModelCreationError("ERROR: Failed to relate trials to a block in the session.")
        # Add block relationship.
        kw['block_id'] = blocks[block_idx].id
        try:
            trial, created = get_one_or_create(CircleTaskTrial, create_func=new_circletask_trial, **kw)
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
    df_hash = md5(df[['df1', 'df2',
                      'df1_grab', 'df1_release',
                      'df2_grab', 'df2_release']].round(5).values.copy(order='C')).hexdigest()
    if df_hash != sent_hash:
        raise UploadError("ERROR: Data corrupted.")
    else:
        return True


def check_circletask_touched(df):
    """ Check if all sliders were used.

    :param df: Dataframe to check.
    :type df: pandas.DataFrame
    :return: Whether all sliders where used.
    :rtype: bool
    """
    #
    untouched = df.isna().all().any()
    if untouched:
        raise UploadError("ERROR: Task not properly executed.")
    return True


def check_circletask_integrity(df, sent_hash):
    """ Evaluate if this data was tampered with and the controls actually touched.

    :param df: Dataframe to check.
    :type df: pandas.DataFrame
    :param sent_hash: The received hash to compare df to.
    :type sent_hash: str
    :return: Status of integrity.
    :rtype: bool
    """
    try:
        check = check_circletask_touched(df) and check_circletask_hash(df, sent_hash)
    except UploadError:
        raise UploadError("ERROR: Data corrupted.")
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
    except IOError:
        raise UploadError("ERROR: Failed to read file contents for user.")
    except IndexError:
        raise UploadError("ERROR: No data in User table.")
    return props


def get_blocks_df(csv_file, session_uid, user_id):
    """ Return a DataFrame from CSV file or buffer and add user_id column.
    
    :param csv_file: Table with session data for blocks.
    :type csv_file: str|io.StringIO
    :param session_uid: Identifier to group blocks as belonging to the same session.
    :type session_uid: str
    :param user_id: ID of the user who performed the session.
    :type user_id: str
    :return: Properties of the blocks.
    :rtype: pandas.DataFrame
    """
    try:
        # Read data and keep empty string in treatment as empty string, not NaN.
        blocks_df = pd.read_csv(csv_file, keep_default_na=False)
    except Exception:
        raise UploadError("ERROR: Failed to read file contents for session.")
    try:
        # Convert time_iso string to datetime.
        blocks_df['time_iso'] = blocks_df['time_iso'].apply(lambda t: datetime.strptime(t, time_fmt))
        blocks_df['user_id'] = user_id
        # Rename 'block' column to CircleTaskBlock model compatible 'nth_block'
        blocks_df.rename(columns={'block': 'nth_block'}, inplace=True)
    except KeyError:
        raise UploadError("ERROR: Missing columns in session data.")
    # Unique identifier for session so we can associate the blocks with this particular session.
    blocks_df['session_uid'] = session_uid
    return blocks_df


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
    """ Read all uploaded blocks with trials, check integrity, concatenate.
    
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
    :return: Properties of all the trials as generator.
    :rtype: generator
    """
    # Look for files containing trial data.
    file_indices = get_file_indices(filenames, 'trials')
    if not file_indices:
        raise UploadError("ERROR: Trial files are missing.")
    
    # Collect data from each file as separate DataFrames.
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
        # Determine to which block in the session this file belongs to.
        mask = (times == trials_meta.time_iso) & (blocks == trials_meta.block)
        try:
            block_idx = mask[mask].index[0]  # The accompanying row in session file.
        except IndexError:
            raise UploadError("ERROR: Mismatch between data in session file and trials file.")
        # Check data integrity by comparing hash values of this file and what was sent with the session file.
        sent_hash = hashes.iloc[block_idx]
        try:
            check_passed = check_circletask_integrity(df, sent_hash)
        except UploadError:
            raise
        # Add block index to relate trials to a CircleTaskBlock object when adding them to the database later on.
        df['block_idx'] = block_idx
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
        
        'blocks': list
        
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
        device_file_idx = get_table_idx(list_of_filenames, 'device')
        user_file_idx = get_table_idx(list_of_filenames, 'user')
        blocks_file_idx = get_table_idx(list_of_filenames, 'session')
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
        kw['device'] = get_device_properties(io.StringIO(decoded_list[device_file_idx]))
        kw['user'] = get_user_properties(io.StringIO(decoded_list[user_file_idx]))
    except UploadError:
        raise
    # Use the time of the session to create a uid from it, so we can group the blocks together later on.
    basename, ext = list_of_filenames[blocks_file_idx].split('.')
    try:
        # Unique id based on this user at that time.
        session_uid = md5((kw['user']['id'] + basename.split('-')[1]).encode()).hexdigest()
    except KeyError:
        raise UploadError("ERROR: User ID is missing.")
    except IndexError:
        raise UploadError("ERROR: Session file name is missing datetime.")
    
    try:
        blocks_df = get_blocks_df(io.StringIO(decoded_list[blocks_file_idx]), session_uid, kw['user']['id'])
        kw['blocks'] = [data._asdict() for data in blocks_df.itertuples(index=False)]  # Convert namedtuple to dict.
        
        kw['trials'] = get_trials_properties(list_of_filenames,
                                             decoded_list,
                                             blocks_df['time_iso'],
                                             blocks_df['nth_block'],
                                             blocks_df['hash'],
                                             kw['user']['id'])
    except (UploadError, KeyError):
        raise UploadError("ERROR: Failed to parse data.")
    
    return kw


def process_upload(filenames, contents):
    """ First parse the uploaded data and then add it to the database.
    
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
        add_to_db(kw['device'], kw['user'], kw['blocks'], kw['trials'])
    except ModelCreationError:
        raise


def records_to_df(store, columns=None):
    """ Convert records-style data to pandas DataFrame.
    
    :param store: Data from a dash_core_components.store component.
    :type store: list[dict]
    :param columns: Columns to use for empty DataFrame in case there are no records.
    :type columns: list[dict]
    :return: Stored data as a DataFrame.
    :rtype: pandas.DataFrame
    """
    df = pd.DataFrame(store)
    # If the DataFrame is practically empty, delete everything except for the columns.
    if df.isna().all().all():
        df = df[0:0]
    if df.columns.empty and columns:
        df = pd.DataFrame(None, columns=[c['id'] for c in columns])
    return df


def df_to_records(df):
    """ Convert pandas DataFrame to table compatible data aka records. If DataFrame is empty keep the columns.
    
    :type df: pandas.DataFrame
    :rtype: list[dict]
    """
    if df.empty:
        # Return column names in 'records' style.
        return [{c: None for c in df.columns}]
    return df.to_dict('records')


################
# UI Callbacks #
################
def register_callbacks(dashapp):
    """ Defines all callbacks to UI events in the dashboard. """
    @dashapp.callback(Output('output-data-upload', 'children'),
                      [Input('upload-data', 'contents')],
                      [State('upload-data', 'filename'),
                       State('upload-data', 'last_modified')])
    def on_upload(list_of_contents, list_of_names, list_of_dates):
        """ Upload data to SQL DB. """
        if list_of_contents is not None:
            try:
                # Insert data into database.
                process_upload(list_of_names, list_of_contents)
            except (UploadError, ModelCreationError) as e:
                # Display the error message.
                return [html.Div(str(e))]
            # Display success message.
            return [html.Div("Upload successful.")]
    
    @dashapp.callback([Output('datastore', 'data'),
                       Output('user-IDs', 'options'),
                       Output('removal-hint', 'children')],
                      [Input('output-data-upload', 'children'),
                       Input('refresh-btn', 'n_clicks'),
                       Input('date-picker-range', 'start_date'),
                       Input('date-picker-range', 'end_date'),
                       ])
    def update_datastore(upload_msg, refresh_clicks, start_date, end_date):
        """ Get data from SQL DB and store in memory.
            Update dropdown options.
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            comp_id = None
        else:
            # Which component triggered the callback?
            comp_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if comp_id == 'output-data-upload':  # When uploading manually on the website.
            try:
                # Query db on initial call when upload_msg is None or on successful upload.
                if upload_msg is None or "Upload successful." in upload_msg[0].children:
                    df = analysis.get_data(start_date, end_date)
                else:
                    return (dash.no_update,) * 3
            except (TypeError, AttributeError, IndexError):
                return (dash.no_update,) * 3
        else:
            df = analysis.get_data(start_date, end_date)
        # Remove invalid trials.
        df_adjusted = analysis.get_valid_trials(df)
        n_removed = len(df) - len(df_adjusted)
        removal_msg = f"{n_removed} trials have been removed from the selected time period due to incorrect execution."\
                      + bool(n_removed) * " Sliders were either not used concurrently or not used at all."
        users = [{'label': p, 'value': p} for p in df['user'].unique()]
        return df_to_records(df_adjusted), users, removal_msg
    
    @dashapp.callback([Output('trials-table', 'data'),
                       Output('trials-table', 'columns'),
                       Output('contour-store', 'data')],
                      [Input('datastore', 'data'),
                       Input('user-IDs', 'value'),
                       Input('contamination', 'value')])
    def set_table(stored_data, users_selected, contamination):
        """ Prepares stored data for display in the main table. Assesses outliers as well. """
        df = records_to_df(stored_data)
        # Get outlier data.
        if not contamination:
            contamination = 0.1
        try:
            outliers, z = analysis.get_outlyingness(df[['df1', 'df2']].values, contamination=contamination)
        except (KeyError, ValueError):
            print("ERROR: Could not compute outliers. Missing columns in DataFrame.")
            # Create data with no outliers.
            outliers = np.array(False).repeat(df.shape[0])
            z = np.ones((101, 101)).astype(int)
        df['outlier'] = outliers.astype(int)
        # Format table columns.
        columns = layout.get_columns_settings(df)
        
        if not users_selected:
            # Return all the rows on initial load/no user selected.
            return df_to_records(df), columns, z.tolist()
        
        df[['user', 'block', 'constraint', 'outlier']] = df[['user', 'block', 'constraint', 'outlier']].astype(
            'category')
        filtered = df.query('`user` in @users_selected')
        return df_to_records(filtered), columns, z.tolist()
    
    @dashapp.callback(Output('filtered-hint', 'children'),
                      [Input('trials-table', 'derived_virtual_data')],
                      [State('trials-table', 'data'),
                       State('trials-table', 'filter_query')])
    def on_table_filter(table_data_filtered, table_data, query):
        """ Update message about removed trials by filtering. """
        # Check if there even is data. The empty dataset has 1 row with all None.
        try:
            if len(table_data) == 1 and np.isnan(np.array(tuple(table_data[0].values()), dtype=np.float)).all():
                n_filtered = 0
            else:
                n_filtered = len(table_data) - len(table_data_filtered)
        except (TypeError, AttributeError):
            n_filtered = 0
        try:
            filter = query.replace('{', '').replace('}', '')
        except AttributeError:
            filter = ""
        filtered_msg = bool(n_filtered) * f" {n_filtered} trials were removed by filters set in the table ({filter})."
        return filtered_msg
    
    @dashapp.callback(Output('pca-store', 'data'),
                      [Input('trials-table', 'derived_virtual_data')],
                      [State('trials-table', 'columns')])
    def set_pca_store(table_data, table_columns):
        """ Save results of PCA into a store. """
        df = records_to_df(table_data, columns=table_columns)
        try:
            df[['user', 'block', 'constraint']] = df[['user', 'block', 'constraint']].astype('category')
            pca_df = df.groupby('block').apply(analysis.get_pca_data)
        except KeyError:
            pca_df = pd.DataFrame()
        else:
            pca_df.reset_index(inplace=True)
            
        if pca_df.empty:
            pca_df = pd.DataFrame(None, columns=['block', 'PC', 'var_expl', 'x', 'y', 'meanx', 'meany'])
        return df_to_records(pca_df)
    
    @dashapp.callback(Output('barplot-pca', 'figure'),
                      [Input('pca-store', 'data')])
    def set_pca_plot(pca_data):
        """ Update bar-plot showing explained variance per principal component. """
        df = records_to_df(pca_data)
        try:
            df[['block', 'PC']] = df[['block', 'PC']].astype('category')
        except KeyError:
            pass
        fig = layout.generate_pca_figure(df)
        return fig
    
    @dashapp.callback([Output('pca-table', 'data'),
                       Output('pca-table', 'columns')],
                      [Input('pca-store', 'data')])
    def set_pca_angle_table(pca_data):
        """ Update table for showing divergence between principal components and UCM vectors. """
        pca_df = records_to_df(pca_data)
        if pca_df.empty:
            angle_df = pd.DataFrame(None, columns=['block', 'PC', 'parallel', 'orthogonal'])
        else:
            ucm_vec = analysis.get_ucm_vec()
            angle_df = pca_df.groupby('block').apply(analysis.get_pc_ucm_angles, ucm_vec)
            angle_df.reset_index(level='block', inplace=True)
        columns = layout.get_pca_columns_settings(angle_df)
        return df_to_records(angle_df), columns
    
    @dashapp.callback([Output('proj-table', 'data'),
                       Output('proj-table', 'columns')],
                      [Input('trials-table', 'derived_virtual_data')])
    def set_proj_table(table_data):
        """ Calculate projections onto UCM parallel and orthogonal vectors and their descriptive statistics and put
         the result into a table.
         """
        df = records_to_df(table_data)
        try:
            df[['user', 'block', 'constraint']] = df[['user', 'block', 'constraint']].astype('category')
        except KeyError:
            raise PreventUpdate
        
        ucm_vec = analysis.get_ucm_vec()
        df_proj = df[['block', 'df1', 'df2']].groupby('block').apply(analysis.get_projections, ucm_vec)
        if df_proj.empty:
            df_proj['parallel'] = np.NaN
            df_proj['orthogonal'] = np.NaN
        df_proj['block'] = df['block']  # FixMe: We need projection length for RM ANOVA. Split function in 2.
        
        # Get statistic characteristics of absolute lengths.
        df_stats = analysis.get_stats(df_proj, by='block')  # FixMe: Avg length, instead of mean, which is 0.
        #df_stats.insert(0, 'projection', df_stats.index)

        # For display in a simple table flatten Multiindex columns.
        df_stats.columns = [" ".join(col).strip() for col in df_stats.columns.to_flat_index()]
        # Get display settings for numeric cells.
        columns = layout.get_columns_settings(df_stats)
        return df_to_records(df_stats), columns
    
    @dashapp.callback(Output('scatterplot-trials', 'figure'),
                      [Input('pca-store', 'data'),  # Delay update until PCA is through.
                       Input('pca-checkbox', 'value')],
                      [State('trials-table', 'derived_virtual_data'),
                       State('datastore', 'data'),
                       State('contour-store', 'data')])
    def on_pca_set_trial_graph(pca_data, show_pca, table_data, stored_data, contour):
        """ Update the graph for displaying trial data as scatter plot. """
        df = records_to_df(table_data)
        z = np.array(contour)
        fig = layout.generate_trials_figure(df, contour_data=z)
        
        # PCA visualisation.
        pca_df = records_to_df(pca_data)
        if 'Show' in show_pca:
            arrows = layout.get_pca_annotations(pca_df)
            fig.layout.update(annotations=arrows)
            
        return fig

    @dashapp.callback([Output('onset-dfs', 'figure'),
                       Output('duration-dfs', 'figure')],
                      [Input('trials-table', 'derived_virtual_data')])
    def on_table_set_violinplot(table_data):
        """ Update histograms when data in trials table changes. """
        df = records_to_df(table_data)
        try:
            fig_onset = layout.generate_grab_figure(df, 'grab')
            fig_duration = layout.generate_grab_figure(df, 'duration')
        except KeyError:
            raise PreventUpdate
        return fig_onset, fig_duration
    
    @dashapp.callback([Output('histogram-dfs', 'figure'),
                       Output('histogram-sum', 'figure')],
                      [Input('trials-table', 'derived_virtual_data')])
    def on_table_set_histograms(table_data):
        """ Update histograms when data in trials table changes. """
        df = records_to_df(table_data)
        try:
            fig_dfs = layout.generate_histograms(df[['df1', 'df2']])
            fig_sum = layout.generate_histograms(df[['sum']])
        except KeyError:
            raise PreventUpdate
        return fig_dfs, fig_sum
    
    @dashapp.callback([Output('corr-table', 'data'),
                       Output('corr-table', 'columns')],
                      [Input('trials-table', 'derived_virtual_data')])
    def set_corr_table(table_data):
        """ Update table showing Pearson correlations between degrees of freedom and their sum. """
        df = records_to_df(table_data)
        try:
            corr = df[['df1', 'df2', 'sum']].corr()
        except KeyError:
            raise PreventUpdate
        
        corr.index.name = ''
        corr.reset_index(inplace=True)
        columns = layout.get_columns_settings(corr)
        return df_to_records(corr), columns
    
    @dashapp.callback([Output('variance-table', 'data'),
                       Output('variance-table', 'columns')],
                      [Input('trials-table', 'derived_virtual_data')])
    def set_variance_table(table_data):
        """ Update table showing variances of dependent and in independent variables. """
        df = records_to_df(table_data)
        try:
            df[['user', 'block', 'constraint']] = df[['user', 'block', 'constraint']].astype('category')
        except KeyError:
            pass
            #raise PreventUpdate
        variances = analysis.get_descriptive_stats(df)
        columns = layout.get_columns_settings(variances)
        return df_to_records(variances), columns
    
    @dashapp.callback(Output('barplot-variance', 'figure'),
                      [Input('variance-table', 'derived_virtual_data')])
    def on_table_set_variance_graph(table_data):
        """ Update graph showing variances of dependent and in independent variables. """
        df = records_to_df(table_data)
        return layout.generate_variance_figure(df)
