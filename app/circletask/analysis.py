from datetime import datetime, timedelta
import itertools

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sqlalchemy import and_

from app.extensions import db
from app.models import CircleTaskBlock, CircleTaskTrial


def get_data(start_date=None, end_date=None):
    """ Queries database for data an merges to a Dataframe. """
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
    # Use users' index instead of id for obfuscation and shorter display.
    users_inv_map = pd.Series(users_df.index, index=users_df.id)
    # Read only trials from blocks we loaded.
    query_stmt = db.session.query(CircleTaskTrial).filter(CircleTaskTrial.block_id.in_(blocks_df.index)).statement
    trials_df = pd.read_sql_query(query_stmt, db.engine, index_col='id')
    # Now insert some data from other tables.
    trials_df.insert(0, 'user', trials_df.user_id.map(users_inv_map))
    trials_df.insert(1, 'session', trials_df['block_id'].map(blocks_df['nth_session']))
    trials_df.insert(2, 'block', trials_df['block_id'].map(blocks_df['nth_block']))
    trials_df.insert(3, 'constraint', trials_df['block_id'].map(blocks_df['treatment']))
    trials_df[['user', 'session', 'block', 'constraint']] = trials_df[['user', 'session',
                                                                       'block', 'constraint']].astype('category')
    # Exclude columns.
    trials_df.drop(columns=['user_id', 'block_id'], inplace=True)
    return trials_df


def get_valid_trials(dataframe):
    """ Remove trials where sliders where not grabbed concurrently or grabbed at all.
    
    :param dataframe: Trial data.
    :type dataframe: pandas.DataFrame
    :return: Filtered trials.
    :rtype: pandas.DataFrame
    """
    # Remove trials with missing values. This means at least one slider wasn't grabbed.
    df = dataframe.dropna(axis='index', how='any')
    # Remove trials where sliders where not grabbed concurrently.
    mask = ~((df['df1_release'] <= df['df2_grab']) | (df['df2_release'] <= df['df1_grab']))
    df = df[mask]
    return df
    
    
def get_descriptive_stats(dataframe):
    """ Get descriptive statistics from trial data.
    
    :param dataframe: Data
    :type dataframe: pandas.Dataframe
    :return: Means, variances for df1, df2 and their sum.
    :rtype: pandas.DataFrame
    """
    vars = ['df1', 'df2', 'sum']
    group_vars = ['user', 'block', 'constraint']
    agg_funcs = ['mean', 'var']
    try:
        grouped = dataframe.groupby(group_vars)
        variances = grouped.agg({v: agg_funcs for v in vars})
    except KeyError:
        raise
    except pd.core.base.DataError:
        # Create empty DataFrame with columns.
        return pd.DataFrame(None, columns=group_vars + [" ".join(i) for i in itertools.product(vars, agg_funcs)])
    
    variances.columns = [" ".join(x) for x in variances.columns.ravel()]
    variances.columns = [x.strip() for x in variances.columns]
    variances.dropna(inplace=True)
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
    :return: mean value, optionally for each group.
    :rtype: numpy.float64|pandas.Series
    """
    if by is None:
        means = dataframe[x].mean()
    else:
        means = dataframe.groupby(by)[x].mean()
    return means


def get_pca_data(dataframe):
    """ Conduct Principal Component Analysis on 2D dataset.
    
    :param dataframe: Data holding 'df1' and 'df2' values as columns.
    :type dataframe: pandas.DataFrame
    :return: Explained variance, components and means.
    :rtype: pandas.DataFrame
    """
    # We don't reduce dimensionality, but overlay the 2 principal components in 2D.
    pca = PCA(n_components=2)
    
    x = dataframe[['df1', 'df2']].values
    try:
        # df1 and df2 have the same scale. No need to standardize. Standardizing might actually distort PCA here.
        pca.fit(x)
    except ValueError:
        # Return empty.
        return pd.DataFrame(columns=['var_expl', 'x', 'y', 'meanx', 'meany'])
    
    df = pd.DataFrame({'var_expl': pca.explained_variance_ratio_.T * 100,  # In percent
                       'x': pca.components_[:, 0],
                       'y': pca.components_[:, 1],
                       'meanx': pca.mean_[0],
                       'meany': pca.mean_[1],
                       },
                      index=[1, 2]  # For designating principal components.
                      )
    return df


def get_pca_vectors(dataframe):
    """ Get principal components for as vectors. Vectors can then be used to annotate graphs.
    
    :param dataframe: Tabular PCA data.
    :type dataframe: pandas.DataFrame
    :return: Principal components as vector pairs in input space with mean as origin first and offset second.
    :rtype: list
    """
    vectors = list()
    # Use the "components" to define the direction of the vectors,
    # and the "explained variance" to define the squared-length of the vectors.
    for pc, row in dataframe.iterrows():
        v = row[['x', 'y']].values * np.sqrt(row['var_expl']) * 5  # Scale up for better visibility.
        mean = row[['meanx', 'meany']].values
        mean_offset = (mean, mean + v)
        vectors.append(mean_offset)
    
    return vectors


def get_pca_vectors_by(dataframe, by=None):
    """ Get principal components for each group as vectors. Vectors can then be used to annotate graphs.

    :param dataframe: Data holding 'df1' and 'df2' values as columns.
    :type dataframe: pandas.DataFrame
    :param by: Column to group data by and return 2 vectors for each group.
    :type by: str|list
    :return: list of principal components as vector pairs in input space with mean as origin first and offset second.
    :rtype: list
    """
    vector_pairs = list()
    if by is None:
        pca_df = get_pca_data(dataframe)
        v = get_pca_vectors(pca_df)
        vector_pairs.append(v)
    else:
        grouped = dataframe.groupby(by)
        for group, data in grouped:
            pca_df = get_pca_data(data)
            v = get_pca_vectors(pca_df)
            vector_pairs.append(v)
            # ToDo: Augment by groupby criteria.
            
    return vector_pairs


def get_interior_angle(vec0, vec1):
    """ Get the smaller angle between vec0 and vec1 in degrees.
    
    :param vec0: Vector 0
    :type vec0: numpy.ndarray
    :param vec1: Vector 1
    :type vec1: numpy.ndarray
    :return: Interior angle between vector0 and vector1 in degrees.
    :rtype: float
    """
    angle = np.math.atan2(np.linalg.det([vec0, vec1]), np.dot(vec0, vec1))
    degrees = abs(np.degrees(angle))
    # Min and max should be between 0° an 90°.
    degrees = min(degrees, 180.0 - degrees)
    return degrees


def get_ucm_vec(p0=None, p1=None):
    """ Returns 2D unit vector in direction of uncontrolled manifold. """
    if p0 is None:
        p0 = np.array([25, 100])
    if p1 is None:
        p1 = np.array([100, 25])
    parallel = p1 - p0
    parallel = parallel / np.linalg.norm(parallel)  # Normalize.
    return parallel


def get_orthogonal_vec2d(vec):
    """ Get a vector that is orthogonal to vec and has same length.
    
    :param vec: 2D Vector
    :return: 2D Vector orthogonal to vec.
    :rtype: numpy.ndarray
    """
    ortho = np.array([-vec[1], vec[0]])
    return ortho


def get_pc_ucm_angles(dataframe, vec_ucm):
    """ Computes the interior angles between pca vectors and ucm parallel/orthogonal vectors.
    
    :param dataframe: PCA data .
    :type dataframe: pandas.DataFrame
    :param vec_ucm: Vector parallel to UCM.
    :type vec_ucm: numpy.ndarray
    :return: Each angle between principal components and UCM parallel and orthogonal vector.
    :rtype: pandas.DataFrame
    """
    vec_ucm_ortho = get_orthogonal_vec2d(vec_ucm)
    df_angles = pd.DataFrame(columns=['parallel', 'orthogonal'], index=dataframe.index)
    for pc, row in dataframe.iterrows():
        angle_parallel = get_interior_angle(vec_ucm, row[['x', 'y']])
        angle_ortho = get_interior_angle(vec_ucm_ortho, row[['x', 'y']])
        df_angles.loc[pc] = [angle_parallel, angle_ortho]
    df_angles[['parallel', 'orthogonal']] = df_angles[['parallel', 'orthogonal']].astype(float)
    df_angles.insert(0, 'pc', df_angles.index + 1)
    return df_angles


def get_outlyingness(data, contamination=0.1):
    """ Outlier detection from covariance estimation in a Gaussian distributed dataset.
    
    :param data: Data in which to detect outliers. Take care that n_samples > n_features ** 2 .
    :type data: pandas.DataFrame
    :param contamination: The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
    Range is (0, 0.5).
    :type contamination: float
    :returns: Decision on each row if it's an outlier. And contour array for drawing ellipse in graph.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    robust_cov = EllipticEnvelope(support_fraction=1., contamination=contamination)
    outlyingness = robust_cov.fit_predict(data)
    decision = (outlyingness-1).astype(bool)
    
    # Visualisation.
    xx, yy = np.meshgrid(np.linspace(0, 100, 101),
                         np.linspace(0, 100, 101))
    z = robust_cov.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    
    return decision, z


def get_projections(points, vec_ucm):
    """ Returns coefficients a and b in x = a*vec_ucm + b*vec_ortho with x being the difference of a data point and
    the mean.
    Projection is computed using a transformation matrix with ucm parallel and orthogonal vectors as basis.
    
    :param points: Data of 2D points.
    :type points: pandas.Dataframe
    :param vec_ucm: Unit vector parallel to uncontrolled manifold.
    :type vec_ucm: numpy.ndarray
    :return: Array with projected lengths onto vector parallel to UCM as 'a', onto vector orthogonal to UCM as 'b'.
    :rtype: pandas.Dataframe
    """
    # Get the vector orthogonal to the UCM.
    vec_ortho = get_orthogonal_vec2d(vec_ucm)
    # Build a transformation matrix with vec_ucm and vec_ortho as new basis vectors.
    A = np.vstack((vec_ucm, vec_ortho)).T
    # Centralize the data.
    diffs = points - points.mean()
    # For computational efficiency we shortcut the calculation with matrix multiplication.
    coeffs = diffs@A
    coeffs.columns = ['parallel', 'orthogonal']
    return coeffs


def get_stats(data, by=None):
    """ Return mean and variance statistics for data.
    
    :param data: numerical data.
    :type data: pandas.Dataframe
    :param by: groupby column name(s)
    :type by: str|List
    :return: Dataframe with columns mean, var, count and column names of data as rows.
    :rtype: pandas.Dataframe
    """
    # When there're no data, return empty DataFrame with columns.
    if data.empty:
        idx = pd.MultiIndex.from_product([data.columns, ['mean', 'var']])
        stats = pd.DataFrame(None, columns=idx)
        stats['count'] = None
        try:
            stats.drop(by, axis='columns', level=0, inplace=True)
            stats.insert(0, by, None)
        except (ValueError, KeyError):
            pass
        return stats
    
    if not by:
        stats = data.agg(['mean', 'var', 'count']).T
        stats['count'] = stats['count'].astype(int)
    else:
        grouped = data.groupby(by)
        stats = grouped.agg(['mean', 'var'])
        stats['count'] = grouped.size()
        stats.reset_index(inplace=True)
    return stats
    
# ToDo: distribution of residuals.
