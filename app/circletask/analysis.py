import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope

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
    trials_df[['user', 'block', 'constraint']] = trials_df[['user', 'block', 'constraint']].astype('category')
    return trials_df


def get_descriptive_stats(dataframe):
    """
    
    :param dataframe: Data
    :type dataframe: pandas.Dataframe
    :return: Means, variances for df1, df2 and their sum.
    """
    grouped = dataframe.groupby(['user', 'block', 'constraint'])
    variances = grouped.agg({'df1': ['mean', 'var'], 'df2': ['mean', 'var'], 'sum': ['mean', 'var']})
    variances.columns = [" ".join(x) for x in variances.columns.ravel()]
    variances.dropna(inplace=True)
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
    :return: mean value, optionally for each group.
    :rtype: numpy.float64|pandas.Series
    """
    if by is None:
        means = dataframe[x].mean()
    else:
        means = dataframe.groupby(by)[x].mean()
    return means


def get_pca_data(dataframe):
    """
    
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
    """
    
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


def get_stats(data):
    """ Return mean and variance statistics for data.
    
    :param data: numerical data.
    :type data: pandas.Dataframe
    :return: Dataframe with columns mean, var, count and column names of data as rows.
    :rtype: pandas.Dataframe
    """
    stats = data.agg(['mean', 'var', 'count']).T
    stats['count'] = stats['count'].astype(int)
    return stats
    
    
# ToDo: Correlation matrix. Note, that there is a reciprocal suppression: r(sum,df1) > 0, r(sum, df2)>0, r(df1,df2)<0
# ToDo: distribution of residuals.
