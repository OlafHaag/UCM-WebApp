"""
This module contains functions for interaction with the database.
"""

from collections import namedtuple
from contextlib import suppress

from functools import wraps

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from psycopg2.extensions import register_adapter, AsIs
from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound

from app.extensions import db
from app.models import CircleTaskBlock, CircleTaskTrial, Device, User
from .exceptions import ModelCreationError

# Numpy data types compatibility with postgresql database.
register_adapter(np.int64, AsIs)
register_adapter(np.float64, AsIs)


####################
#    Decorators    #
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
        except IntegrityError as e:
            try:
                return result(model.query.filter_by(**kwargs).one(), False)
            except NoResultFound:
                raise ModelCreationError(f"{e.orig.args[0]}")


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
                device_id=kwargs['device_id'],
                age_group=kwargs['age_group'],
                gender=kwargs['gender'],
                gaming_exp=kwargs.get('gaming_exp', None),  # Gaming experience might not be measured by all studies.
                )
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
                            hash=kwargs['hash'],
                            rating=kwargs['rating'],
                            )
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
    except ModelCreationError:
        # There seems to be a conflict between incoming and existing device data.
        # Though it's the same device the resolution may change, when the OS topbar and navbar are hidden,
        # or window size changes. Maybe it's just a floating point precision issue.
        # Haven't figured out yet when the app switches to immersive mode.
        # This, of course, bears the risk of changing device statistics when not handled.
        # ToDo: Handle change in device properties. Ideally register as new device, if necessary. 
        #       Or change floating point precision.
        try:
            device = db.session.query(Device).get(device_kwargs['id'])
        except NoResultFound:
            raise ModelCreationError("ERROR: Could neither retrieve, nor create device.")
        # Update age & gender.
        try:
            device.screen_x = device_kwargs['screen_x']
            device.screen_y = device_kwargs['screen_y']
            device.dpi = device_kwargs['dpi']
            device.density = device_kwargs['density']
            device.aspect_ratio = device_kwargs['aspect_ratio']
            device.size_x = device_kwargs['size_x']
            device.size_y = device_kwargs['size_y']
            device.platform = device_kwargs['platform']
        except KeyError:
            raise ModelCreationError("ERROR: Missing device data.")
    
    try:
        user, is_new_user = get_one_or_create(User, create_func=new_user, **user_kwargs)
    except ModelCreationError:
        # There seems to be a conflict between incoming and existing user data.
        # A user could change their age and gender, so we update it.
        # This, of course, bears the risk of changing demographic statistics for other studies,
        # but for now we roll with it.
        # ToDo: Maybe a combined primary key of id, age, gender is the solution for changing user demographic.
        try:
            user = db.session.query(User).get(user_kwargs['id'])
        except NoResultFound:
            raise ModelCreationError("ERROR: Could neither retrieve, nor create user.")
        # Update age & gender.
        try:
            user.age_group = user_kwargs['age_group']
            user.gender = user_kwargs['gender']
            user.gaming_exp = user_kwargs.get('gaming_exp', None)
        except KeyError:
            raise ModelCreationError("ERROR: Missing user data.")
    
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


def get_data(start_date=None, end_date=None):
    """ Queries database and returns Dataframes for users, circletask_blocks and circletask_trials tables.
    Data between given start date and end date is returned. If not set, all data is returned.
    
    :returns: Data queried from tables as DataFrames.
    :rtype: tuple[pandas.DataFrame]
    """
    # Get data in time period.
    if not start_date and not end_date:
        blocks_df = pd.read_sql_table('circletask_blocks', db.engine, index_col='id')
    elif start_date and not end_date:
        query_stmt = db.session.query(CircleTaskBlock).filter(CircleTaskBlock.time_iso >= start_date).statement
        blocks_df = pd.read_sql_query(query_stmt, db.engine, index_col='id')
    else:
        # end_date's daytime is set to the start of the day (00:00:00), but we want the end of the day.
        end_date = datetime.fromisoformat(end_date) + timedelta(days=1)
        query_stmt = db.session.query(CircleTaskBlock).filter(and_(CircleTaskBlock.time_iso >= start_date,
                                                                   CircleTaskBlock.time_iso < end_date)).statement
        blocks_df = pd.read_sql_query(query_stmt, db.engine, index_col='id')
    
    users_df = pd.read_sql_table('users', db.engine)
    # Read only trials from blocks we loaded.
    query_stmt = db.session.query(CircleTaskTrial).filter(CircleTaskTrial.block_id.in_(blocks_df.index)).statement
    trials_df = pd.read_sql_query(query_stmt, db.engine, index_col='id')
    return users_df, blocks_df, trials_df
