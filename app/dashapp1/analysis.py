import pandas as pd
import numpy as np

from app.extensions import db


def get_data():
    """ Queries database for data an merges to a Dataframe. """
    # This is considered temporary.
    trials_df = pd.read_sql_table('circle_tasks', db.engine)
    user_df = pd.read_sql_table('users', db.engine)
    # Use users index instead of id for obfuscation.
    user_inv_map = {v: k for k, v in user_df.id.iteritems()}
    trials_df.insert(0, 'user', trials_df.user_id.map(user_inv_map))
    # Get blocks.
    session_df = pd.read_sql_table('ct_sessions', db.engine)
    block_map = dict(session_df[['id', 'block']].to_dict('split')['data'])
    trials_df.insert(1, 'block', trials_df['session'].map(block_map))
    treatment_map = dict(session_df[['id', 'treatment']].to_dict('split')['data'])
    trials_df.insert(2, 'constraint', trials_df['session'].map(treatment_map))
    # Exclude columns.
    trials_df.drop(columns=['id', 'user_id', 'session'], inplace=True)
    return trials_df
