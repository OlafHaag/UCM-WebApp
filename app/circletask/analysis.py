import itertools

import pandas as pd
import pingouin as pg
import numpy as np
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope


def join_data(users, blocks, trials):
    """ Take data from different database tables and join them to a single DataFrame. Some variables are renamed and
    recoded in the process, some are dropped.
    
    :param users: Data from users table
    :type users: pandas.DataFrame
    :param blocks: Data from circletask_blocks table.
    :type blocks: pandas.DataFrame
    :param trials: Data from circletask_trials table.
    :type trials: pandas.DataFrame
    :return: Joined and recoded DataFrame.
    :rtype: pandas.DataFrame
    """
    # Use users' index instead of id for obfuscation and shorter display.
    users_inv_map = pd.Series(users.index, index=users.id)
    # Now insert some data from other tables.
    df = pd.DataFrame(index=trials.index)
    df['user'] = trials.user_id.map(users_inv_map).astype('category')
    df['session'] = trials['block_id'].map(blocks['nth_session']).astype('category')
    # Map whole sessions to the constraint in the treatment block as a condition for easier grouping during analysis.
    df['condition'] = trials['block_id'].map(blocks[['session_uid', 'treatment']].replace(
        {'treatment': {'': np.nan}}).groupby('session_uid')['treatment'].ffill().bfill()).astype('category')
    df['block'] = trials['block_id'].map(blocks['nth_block']).astype('category')
    df['treatment'] = trials['block_id'].map(blocks.treatment.where(blocks.treatment != '',
                                                                    blocks.nth_block.map({1: 'pre', 3: 'post'}))
                                             ).astype('category')
    # Add pre-treatment and post-treatment labels to trials for each block.
    # Theoretically, one could have changed number of blocks and order of treatment, but we assume default order here.
    #df['treatment'] = trials['block_id'].map(blocks.treatment.replace(to_replace={r'\w+': 1, r'^\s*$': 0},
    #                                                                  regex=True)).astype('category')
    df['constraint'] = trials['block_id'].map(blocks['treatment']).astype('category')  # Obsolete.
    
    df = pd.concat((df, trials), axis='columns')
    # Add columns for easier filtering.
    df['grab_diff'] = (df['df2_grab'] - df['df1_grab']).abs()
    df['duration_diff'] = (df['df2_duration'] - df['df1_duration']).abs()
    # Exclude columns.
    df.drop(columns=['user_id', 'block_id'], inplace=True)
    return df


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
        df = pd.DataFrame(columns=['var_expl', 'var_expl_ratio', 'x', 'y', 'meanx', 'meany'])
    else:
        df = pd.DataFrame({'var_expl': pca.explained_variance_.T,
                           'var_expl_ratio': pca.explained_variance_ratio_.T * 100,  # In percent
                           'x': pca.components_[:, 0],
                           'y': pca.components_[:, 1],
                           'meanx': pca.mean_[0],
                           'meany': pca.mean_[1],
                           },
                          index=[1, 2]  # For designating principal components.
                          )
    df.index.rename('PC', inplace=True)
    return df


def get_pca_vectors(dataframe):
    """ Get principal components as vectors. Vectors can then be used to annotate graphs.
    
    :param dataframe: Tabular PCA data.
    :type dataframe: pandas.DataFrame
    :return: Principal components as vector pairs in input space with mean as origin first and offset second.
    :rtype: list
    """
    vectors = list()
    # Use the "components" to define the direction of the vectors,
    # and the "explained variance" to define the squared-length of the vectors.
    for idx, row in dataframe.iterrows():
        v = row[['x', 'y']].values * np.sqrt(row['var_expl']) * 3  # Scale up for better visibility.
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
    df_angles = pd.DataFrame(columns=['parallel', 'orthogonal'])
    for idx, row in dataframe.iterrows():
        angle_parallel = get_interior_angle(vec_ucm, row[['x', 'y']])
        angle_ortho = get_interior_angle(vec_ucm_ortho, row[['x', 'y']])
        df_angles.loc[idx] = [angle_parallel, angle_ortho]
    df_angles[['parallel', 'orthogonal']] = df_angles[['parallel', 'orthogonal']].astype(float)
    df_angles.insert(0, 'PC', dataframe['PC'])
    return df_angles


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
    A = np.vstack((vec_ucm, vec_ortho)).T  # A is not an orthogonal projection matrix (A=A.T), but this works.
    # Centralize the data. Analogous to calculating across trials deviation from average for each time step.
    diffs = points - points.mean()
    # For computational efficiency we shortcut the projection calculation with matrix multiplication.
    # The actual math behind it:
    #   coeffs = vec_ucm.T@diff/np.sqrt(vec_ucm.T@vec_ucm), vec_ortho.T@diff/np.sqrt(vec_ortho.T@vec_ortho)
    # Biased variance (normalized by (n-1)) of projection onto UCM vector:
    #   var_ucm = vec_ucm.T@np.cov(diffs, bias=True, rowvar=False)@vec_ucm/(vec_ucm.T@vec_ucm)  # Rayleigh fraction.
    coeffs = diffs@A
    coeffs.columns = ['parallel', 'orthogonal']
    return coeffs


def get_synergy_indices(variances, n=2, d=1):
    """
    n: Number of degrees of freedom. In our case 2.
    
    d: Dimensionality of performance variable. In our case a scalar (1).
    
    Vucm = 1/N * 1/(n - d) * sum(ProjUCM**2)
    
    Vort = 1/N * 1/(d) * sum(ProjORT**2)
    
    Vtotal = 1/n * (d * Vort + (n-d) * Vucm)  # Anull the weights on Vucm and Vort for the sum.
    
    dV = (Vucm - Vort) / Vtotal
    dV = n*(Vucm - Vort) / ((n - d)*Vucm + d*Vort)
    
    Zhang (2008) without weighting Vucm, Vort and Vtotal first:
    dV = n * (Vucm/(n - d) - Vort/d) / (Vucm + Vort)
    
    dVz = 0.5*ln((n / d + dV) / (n / ((n - d) - dV))
    dVz = 0.5*ln((2 + dV) / (2 - dV))
    Reference: https://www.frontiersin.org/articles/10.3389/fnagi.2019.00032/full#supplementary-material
    
    :param variances: Unweighted variances of parallel and orthogonal projections to the UCM.
    :type variances: pandas.DataFrame
    :param n: Number of degrees of freedom. Defaults to 2.
    :type: int
    :param d: Dimensionality of performance variable. Defaults to 1.
    :type d: int
    :returns: Synergy index, Fisher's z-transformed synergy index.
    :rtype: pandas.DataFrame
    """
    try:
        dV = n * (variances['parallel']/(n-d) - variances['orthogonal']/d) \
             / variances[['parallel', 'orthogonal']].sum(axis='columns')
    except KeyError:
        synergy_indices = pd.DataFrame(columns=["dV", "dVz"])
    else:
        dVz = 0.5 * np.log((n/d + dV)/(n/(n-d) - dV))
        synergy_indices = pd.DataFrame({"dV": dV, "dVz": dVz})
    return synergy_indices


def get_synergy_idx_bounds(n=2, d=1):
    """ Get lower and upper bounds of the synergy index.
    
     dV = n * (Vucm/(n - d) - Vort/d) / (Vucm + Vort)
     
     If all variance lies within the UCM, then Vort=0 and it follows for the upper bound: dV = n/(n-d)
     
     If all variance lies within Vort, then Vucm=0 and it follows for the lower bound: dV = -n/d
    
    :param n: Number of degrees of freedom.
    :type: int
    :param d: Dimensionality of performance variable.
    :type d: int
    :returns: Lower and upper bounds of synergy index.
    :rtype: tuple
    """
    dV_lower = -n/d
    dV_upper = n/(n-d)
    return dV_lower, dV_upper
    
    
def get_mean(dataframe, column, by=None):
    """ Return mean values of column x (optionally grouped)
    
    :param dataframe: Data
    :type dataframe: pandas.Dataframe
    :param column: Column name
    :type column: str
    :param by: Column names by which to group.
    :type by: str|list
    :return: mean value, optionally for each group.
    :rtype: numpy.float64|pandas.Series
    """
    if by is None:
        means = dataframe[column].mean()
    else:
        means = dataframe.groupby(by)[column].mean()
    return means


def get_descriptive_stats(data, by=None):
    """ Return mean and variance statistics for data.
    
    :param data: numerical data.
    :type data: pandas.Dataframe
    :param by: groupby column name(s)
    :type by: str|List
    :return: Dataframe with columns mean, var, count and column names of data as rows.
    :rtype: pandas.Dataframe
    """
    # There's a bug in pandas 1.0.4 where you can't use custom numpy functions in agg anymore (ValueError).
    # Note that the variance of projections is usually divided by (n-d) for Vucm and d for Vort. Both are 1 in our case.
    # Pandas default var returns unbiased population variance /(n-1). Doesn't make a difference for synergy indices.
    f_var = lambda series: series.var(ddof=0)
    f_var.__name__ = 'variance'  # Column name gets function name.
    f_avg = lambda series: series.abs().mean()
    f_avg.__name__ = 'absolute average'
    # When there're no data, return empty DataFrame with columns.
    if data.empty:
        if by:
            data.set_index(by, drop=True, inplace=True)
        col_idx = pd.MultiIndex.from_product([data.columns, [f_avg.__name__, 'mean', f_var.__name__]])
        stats = pd.DataFrame(None, index=data.index, columns=col_idx)
        stats['count'] = None
        return stats
    
    if not by:
        stats = data.agg([f_avg, 'mean', f_var, 'count']).T
        stats['count'] = stats['count'].astype(int)
    else:
        grouped = data.groupby(by)
        stats = grouped.agg([f_avg, 'mean', f_var])
        stats['count'] = grouped.size()
        stats.dropna(inplace=True)
    return stats


def get_statistics(df_trials, df_proj):
    """ Calculate descriptive statistics for key values of the anaylsis.
    
    :param df_trials: Data from joined table on trials.
    :type df_trials: pandas.DataFrame
    :param df_proj: Projections onto UCM and its orthogonal space.
    :type df_proj: pandas.DataFrame
    :return: Descriptive statistics and synergy indices.
    :rtype: pandas.DataFrame
    """
    groupers = ['user', 'session', 'condition', 'block', 'treatment']
    try:
        # Get only those trials we have the projections for, in the same order.
        df_trials = df_trials.iloc[df_proj.index]
        df_trials[groupers] = df_trials[groupers].astype('category')
    except (KeyError, ValueError):
        df_proj_stats = get_descriptive_stats(pd.DataFrame(columns=df_proj.columns))
        df_dof_stats = get_descriptive_stats(pd.DataFrame(columns=df_trials.columns))
        cov = pd.DataFrame(columns=[('df1,df2 covariance', '')])
    else:
        df_proj[groupers] = df_trials[groupers]
        # Get statistic characteristics of absolute lengths.
        df_proj_stats = get_descriptive_stats(df_proj, by=groupers)
        # Clean-up to match data on degrees of freedom.
        df_proj_stats.dropna(inplace=True)
        df_dof_stats = get_descriptive_stats(df_trials[groupers + ['df1', 'df2', 'sum']], by=groupers)
        # For degrees of freedom absolute average is the same as the mean, since there are no negative values.
        df_dof_stats.drop('absolute average', axis='columns', level=1, inplace=True)
        # Get covariance between degrees of freedom.
        cov = df_trials.groupby(groupers).apply(lambda x: np.cov(x[['df1', 'df2']].T, ddof=0)[0, 1])
        try:
            cov = cov.to_frame(('df1,df2 covariance', ''))  # MultiIndex.
        except AttributeError:  # In case cov is an empty Dataframe.
            cov = pd.DataFrame(columns=pd.MultiIndex.from_tuples([('df1,df2 covariance', '')]))

    # We now have 1 count column too many, since the projection statistics already has the identical column.
    df_dof_stats.drop('count', axis='columns', level=0, inplace=True)
    # For projections the mean is 0, since projections are from deviations from the mean. So we don't need to show it.
    df_proj_stats.drop('mean', axis='columns', level=1, inplace=True)
    # Get synergy indices based on projection variances we just calculated.
    df_synergies = get_synergy_indices(df_proj_stats.xs('variance', level=1, axis='columns'))
    # Before we merge dataframes, give this one a Multiindex, too.
    df_synergies.columns = pd.MultiIndex.from_product([df_synergies.columns, ['']])
    # Join the 3 statistics to be displayed in a single table.
    df = pd.concat((df_dof_stats, cov, df_proj_stats, df_synergies), axis='columns')
    return df


def wilcoxon_rank_test(data):
    w, p = wilcoxon(data['parallel'], data['orthogonal'], alternative='greater')
    return p < 0.05, w, p


def wide_to_long(df, stubs, suffixes, j):
    """ Transforms a dataframe to long format, where the stubs are melted into a single column with name j and suffixes
    into value columns. Filters for all columns that are a stubs+suffixes combination.
    Keeps 'user', 'treatment' as id_vars. When an error is encountered an emtpy dataframe is returned.
    
    :param df: Data in wide/mixed format.
    :type df: pandas.DataFrame
    :param stubs: First part of a column name. These names will be the values of the new column j.
    :type stubs: list[str]
    :param suffixes: Second part of a column name. These will be the new columns holding the respective values.
    :type suffixes: str|list[str]
    :param j: Name for new column containing stubs.
    :type j: str
    
    :return: Filtered Dataframe in long format.
    :rtype: pandas.Dataframe
    """
    if isinstance(suffixes, str):
        suffixes = [suffixes]
    # We want all stubs+suffix combinations as columns.
    val_cols = [" ".join(x) for x in itertools.product(stubs, suffixes)]
    try:
        # Filter for data we want to plot.
        df = df[['user', 'condition', 'block', 'treatment', *val_cols]]
        # Reverse stub and suffix for long format. We want the measurements as columns, not the categories.
        df.columns = [" ".join(x.split(" ")[::-1]) for x in df.columns]
        long_df = pd.wide_to_long(df=df, stubnames=suffixes, i=['user', 'condition', 'block', 'treatment'],
                                  j=j, sep=" ", suffix=f'(!?{"|".join(stubs)})')
        long_df.reset_index(inplace=True)
    except (KeyError, ValueError):
        long_df = pd.DataFrame(columns=['user', 'condition', 'block', 'treatment', j, *suffixes])
    long_df[['user', 'condition', 'block', 'treatment']] = long_df[['user', 'condition',
                                                                    'block', 'treatment']].astype('category')
    return long_df


def mixed_anova_synergy_index_z(dataframe):
    """ 3 x (3) Two-way split-plot ANOVA with between-factor condition and within-factor block.
    
    :param dataframe: Aggregated data containing Fisher-z-transformed synergy index.
    :type dataframe: pandas.DataFrame
    :return: mixed-design ANOVA results.
    :rtype: pandas.DataFrame
    """
    aov = pg.mixed_anova(data=dataframe, dv='dVz', within='block', subject='user', between='condition')
    return aov
