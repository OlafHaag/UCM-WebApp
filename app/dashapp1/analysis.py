import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

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


def get_descriptive_stats(dataframe):
    grouped = dataframe.groupby(['user', 'block', 'constraint'])
    variances = grouped.agg({'df1': ['mean', 'var'], 'df2': ['mean', 'var'], 'sum': ['mean', 'var']})
    variances.columns = [" ".join(x) for x in variances.columns.ravel()]
    variances.columns = [x.strip() for x in variances.columns]
    variances.reset_index(inplace=True)
    return variances


def get_mean_x_by(dataframe, x, by=None):
    """ Return mean values of column x (optionally grouped)
    :param dataframe: Data
    :type dataframe: pandas.Dataframe
    :param x: Column name
    :type x: str
    :param by: Column names by which to group.
    :type by: str|list
    :return:
    """
    if by is None:
        means = dataframe[x].mean()
    else:
        means = dataframe.groupby(by)[x].mean()
    return means


def get_pca_vectors(dataframe):
    """
    
    :param dataframe: Data holding 'df1' and 'df2' values as columns.
    :type dataframe: pandas.DataFrame
    :return: principal components as vector pairs in input space with mean as origin first and offset second.
    :rtype: list|None
    """
    # We don't reduce dimensionality, but overlay the 2 principal components in 2D.
    pca = PCA(n_components=2)
    
    x = dataframe[['df1', 'df2']].values
    try:
        # df1 and df2 have the same scale. No need to standardize.
        pca.fit(x)
    except ValueError:
        return None
    
    vectors = list()
    # Use the "components" to define the direction of the vectors,
    # and the "explained variance" to define the squared-length of the vectors.
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)
        mean_offset = (pca.mean_, pca.mean_ + v)
        vectors.append(mean_offset)
    
    return vectors


def get_pca_vectors_by(dataframe, by=None):
    """

    :param dataframe: Data holding 'df1' and 'df2' values as columns.
    :type dataframe: pandas.DataFrame
    :param by: Column to group data by and return 2 vectors for each group.
    :type by: str|list
    :return: list of principal components as vector pairs in input space with mean as origin first and offset second.
    :rtype: list
    """
    vector_pairs = list()
    if by is None:
        v = get_pca_vectors(dataframe)
        vector_pairs.append(v)
    else:
        grouped = dataframe.groupby(by)
        for group, data in grouped:
            x = data[['df1', 'df2']]
            v = get_pca_vectors(x)
            vector_pairs.append(v)
            
    return vector_pairs
