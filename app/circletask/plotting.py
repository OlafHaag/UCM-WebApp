"""
This module contains functions for creating figures.
"""
import string

import numpy as np
from plotly import express as px, graph_objs as go, figure_factory as ff
from plotly.subplots import make_subplots

from . import analysis

theme = {'graph_margins': {'l': 40, 'b': 40, 't': 40, 'r': 10},
         # Use colors consistently to quickly grasp what is what.
         'df1': 'cornflowerblue',
         'df2': 'palevioletred',
         'sum': 'peru',
         'parallel': 'lightgreen',
         'orthogonal': 'salmon',
         'colors': px.colors.qualitative.Plotly,
         }


def get_ellipse_coordinates(x_center=0, y_center=0, axis=(1, 0), a=1, b=1, n=100):
    """ Helper function to generate coordinates for drawing an ellipse.
    
    :param x_center: X coordinates of ellipse center.
    :param y_center: Y coordinates of ellipse center.
    :param axis: Ellipse main axis direction.
    :param a: Ellipse parameter for axis.
    :param b: Ellipse parameter for orthogonal axis.
    :param n: Resolution.
    :return: X and y path coordinates for ellipse. Plot with scatter and mode set to lines.
    """
    # We work with unit vectors.
    if not np.isclose(np.linalg.norm(axis), 1):
        axis = np.array(axis)/np.linalg.norm(axis)
    axis2 = analysis.get_orthogonal_vec2d(axis)
    t = np.linspace(0, 2 * np.pi, n)
    # Ellipse parameterization with respect to a system of axes of directions defined by axis.
    xs = a * np.cos(t)
    ys = b * np.sin(t)
    # Construct rotation matrix.
    rot_mat = np.array([axis, axis2]).T
    # Coordinates of the points with respect to the cartesian system with basis vectors [1, 0], [0,1].
    xp, yp = np.dot(rot_mat, [xs, ys])
    # Move to center coordinates.
    x = xp + x_center
    y = yp + y_center
    return x, y


def generate_trials_figure(df, contour_data=None):
    """ Scatter plot of data.
    
    :param df: Data
    :type df: pandas.DataFrame
    :param contour_data: outlier visualisation data.
    :type: numpy.ndarray
    :rtype: plotly.graph_objs.figure
    """
    if df.empty:
        data = []
    else:
        data = [go.Scattergl(
            x=df[df['block'] == i]['df1'],
            y=df[df['block'] == i]['df2'],
            text=[f"df1={j['df1']:.2f}<br />df2={j['df2']:.2f}<br />Sum={j['sum']:.2f}<br />Participant {j['user']}"
                  f"<br />Session {j['session']}<br />Block {i}<br />Trial {j['trial']}"
                  for _, j in df[df['block'] == i].iterrows()],
            hoverinfo='text',
            mode='markers',
            opacity=0.7,
            marker={'size': 10,
                    'color': theme['colors'][i],
                    'line': {'width': 0.5, 'color': 'white'}
                    },
            name=f"Block {i} {','.join(df[df['block'] == i]['constraint'].unique())}",
        ) for i in df['block'].unique()]

    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
    )
    
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            xaxis={'title': 'Degree of Freedom 1', 'range': [0, 100], 'constrain': 'domain'},
            yaxis={'title': 'Degree of Freedom 2', 'range': [0, 100], 'scaleanchor': 'x', 'scaleratio': 1},
            margin=theme['graph_margins'],
            legend=legend,
            hovermode='closest',
        )
    )
    # Task goal 1 visualization.
    fig.add_trace(go.Scatter(
        x=[25, 100],
        y=[100, 25],
        mode='lines',
        name="task goal 1",
        marker={'color': 'black',
                },
        hovertemplate="df1+df2=125",
    ))
    # Task goal 2 (DoF constrained) visualization.
    fig.add_scatter(y=[62.5], x=[62.5],
                    name="task goal 2",
                    text=["task goal 2 (df1=df2)"],
                    hoverinfo='text',
                    mode='markers',
                    marker_symbol='x',
                    opacity=0.7,
                    marker={'size': 25,
                            'color': 'red'})

    # Add visualisation for outlier detection.
    if isinstance(contour_data, np.ndarray):
        fig.add_trace(go.Contour(z=contour_data,
                                 name='outlier threshold',
                                 line_color='black',
                                 contours_type='constraint',
                                 contours_operation='=',
                                 contours_value=-1,
                                 contours_showlines=False,
                                 line_width=1,
                                 opacity=0.25,
                                 showscale=False,
                                 showlegend=True,
                                 hoverinfo='skip',
                                 ))

    fig.update_xaxes(hoverformat='.2f')
    fig.update_yaxes(hoverformat='.2f')

    return fig


def get_pca_annotations(pca_dataframe):
    """ Generate display properties of principal components for graphs.
    
    :param pca_dataframe: Results of PCA.
    :type pca_dataframe: pandas.DataFrame
    :return: List of properties for displaying arrows.
    :rtype: list
    """
    # Visualize the principal components as vectors over the input data.
    arrows = list()
    # Each block displays its principal components.
    try:
        for name, group in pca_dataframe.groupby('block'):
            vectors = analysis.get_pca_vectors(group)  # origin->destination pairs.
            
            arrows.extend([dict(
                ax=v0[0],
                ay=v0[1],
                axref='x',
                ayref='y',
                x=v1[0],
                y=v1[1],
                xref='x',
                yref='y',
                showarrow=True,
                arrowhead=3,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor=theme['colors'][name]  # Match arrow color to block.
            )
                for v0, v1 in vectors])
    except KeyError:
        pass
    return arrows


def add_pca_ellipses(fig, pca_dataframe):
    """ Get data for drawing ellipses around data for each block.
    Ellipses are only scaled by explained variance, not by spread of the actual data.
    
    :param fig: Figure to add ellipses to.
    :param pca_dataframe: Tabular results of PCA.
    """
    # Each block displays its principal components.
    try:
        for name, group in pca_dataframe.groupby('block'):
            x, y = get_ellipse_coordinates(*group[['meanx', 'meany']].iloc[0],
                                           axis=group[['x', 'y']].iloc[0],
                                           a=np.sqrt(group['var_expl'].iloc[0])*2,
                                           b=np.sqrt(group['var_expl'].iloc[1])*2,
                                           )
            fig.add_scatter(x=x,
                            y=y,
                            mode='lines',
                            line_color=theme['colors'][name],
                            showlegend=False,
                            hoverinfo='skip',
                            )
    except (KeyError, IndexError):
        pass


def generate_histograms(dataframe, by=None):
    """ Plot distribution of data to visually check for normal distribution.
    
    :param dataframe: Data for binning. When by is given this must be only 2 columns with one of them being the grouper.
    :type dataframe: pandas.DataFrame
    :param by: dataframe column to group data by.
    :type by: str
    """
    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
    )
    
    cols = list(dataframe.columns)
    if dataframe.empty:
        fig = go.Figure()
        fig.update_xaxes(range=[0, 100])
    else:
        if not by:
            # Columns we want to plot histograms for. Display order is reversed.
            data = [dataframe[c] for c in dataframe.columns]
            try:
                colors = [theme[c] for c in dataframe.columns]
            except KeyError:
                colors = [theme['colors'][i + 1] for i in range(len(dataframe.colums))]
            # Create distplot with curve_type set to 'normal', overriding default 'kde'.
            fig = ff.create_distplot(data,  dataframe.columns, colors=colors, curve_type='normal')
        else:
            data = list()
            labels = list()
            colors = list()
            grouped = dataframe.groupby(by)
            for i, (name, df) in enumerate(grouped):
                data.append(df.drop(columns=by).squeeze(axis='columns'))
                labels.append(f"{by.capitalize()} {name}")  # Potential risk when 'by' is a list.
                # Set theme colors for traces.
                try:
                    color = theme[name]
                except KeyError:
                    color = theme['colors'][i+1]
                colors.append(color)
            fig = ff.create_distplot(data,  labels, colors=colors, curve_type='normal')  # Override default 'kde'.
            
    fig.layout.update(legend=legend,
                      yaxis={'title': 'Probability Density'},
                      xaxis={'title': 'Final State Value'},
                      margin=theme['graph_margins'],
                      )
    return fig


def generate_pca_figure(dataframe):
    """ Plot explained variance by principal components as Bar plot with cumulative explained variance.
    
    :param dataframe: Results of PCA.
    :type dataframe: pandas.DataFrame
    :return: Properties of plot as a dictionary.
    :rtype: dict
    """
    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
        title={'text': 'PC'},
    )
    
    layout = dict(
        legend=legend,
        yaxis={'title': 'Explained variance in percent'},
        xaxis={'title': 'Block'},
        margin={'l': 60, 'b': 40, 't': 40, 'r': 10},
        hovermode=False,
    )
    
    try:
        fig = px.bar(dataframe, x='block', y='var_expl', barmode='group', color='PC')
        fig.update_xaxes(tickvals=dataframe['block'].unique())
    except (KeyError, ValueError):
        fig = go.Figure()
    else:
        fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    finally:
        fig.layout.update(**layout)
    
    return fig


def generate_means_figure(dataframe, variables=None):
    """ Barplots for variables grouped by block.
    Variable for each user is plotted as well as mean over all users.
    
    :param dataframe: Data with variables.
    :type dataframe: pandas.DataFrame
    :param variables: Variables to plot by block. List of dicts with 'label' and 'var' keys.
    :type variables: list[dict]
    :return: Figure object.
    """
    if not variables:
        variables = [{'label': 'Sum Variance', 'var': 'sum variance'},
                     {'label': 'Sum Mean', 'var': 'sum mean'}]
        
    fig = make_subplots(rows=len(variables), cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.04,
                        row_titles=list(string.ascii_uppercase[:len(variables)]))  # A, B, C, ...
    # Handle empty data.
    if dataframe.empty:
        fig.layout.update(xaxis2_title='Block',
                          margin=theme['graph_margins'],
                          )
        # Empty dummy traces.
        for i, v in enumerate(variables):
            fig.add_trace({}, row=i+1, col=1)
            fig.update_yaxes(title_text=v['label'], hoverformat='.2f', row=i+1, col=1)
        fig.update_xaxes(title_text="Block", row=len(variables), col=1)
        return fig
    
    # Subplots for variables.
    blocks = dataframe['block'].unique()
    grouped = dataframe.groupby('user')
    # Variance plot.
    for i, v in enumerate(variables):
        for name, group in grouped:
            fig.add_trace(go.Bar(
                x=group['block'],
                y=group[v['var']],
                name=f'Participant {name}',
                showlegend=False,
                marker={'color': [theme['colors'][j] for j in group['block']],
                        'opacity': 0.5},
                hovertemplate="%{text}: %{y:.2f}",
                text=[v['label']] * len(group),
            ),
                row=i+1, col=1,
            )
    
        # Add mean across participants by block
        means = analysis.get_mean(dataframe, column=v['var'], by='block')
        for block, value in means.iteritems():
            fig.add_trace(go.Scatter(
                x=[block - 0.5, block, block + 0.5],
                y=[value, value, value],
                name=f"Block {block}",
                showlegend=(not i),  # Show legend only for first trace to prevent duplicates.
                hovertext=f'Block {block}',
                hoverinfo='y',
                textposition='top center',
                mode='lines',
                marker={'color': theme['colors'][block]},
                hovertemplate=f"Mean {v['label']}: {value:.2f}",
            ), row=i+1, col=1)
        fig.update_yaxes(title_text=v['label'], hoverformat='.2f', row=i+1, col=1)
    fig.update_xaxes(tickvals=blocks, title_text="Block", row=len(variables), col=1)

    # Layout
    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
        itemsizing='constant',
        title={'text': 'Mean'},
    )
    
    fig.update_layout(
        barmode='group',
        bargap=0.15,  # Gap between bars of adjacent location coordinates.
        bargroupgap=0.1,  # Gap between bars of the same location coordinate.
        margin=theme['graph_margins'],
        legend=legend,
        hovermode='closest',
    )
    return fig


def generate_violin_figure(dataframe, columns, ytitle):
    """ Plot 2 columns of data as violin plot, grouped by block.

    :param dataframe: Variance of projections.
    :type dataframe: pandas.DataFrame
    :param columns: 2 columns for the negative and the positive side of the violins.
    :type columns: list
    :param ytitle: Title of Y-axis. What is being plotted? What are the units of the data?
    :type ytitle: str

    :return: Figure object of graph.
    :rtype: plotly.graph_objs.Figure
    """
    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
    )
    
    fig = go.Figure()
    fig.layout.update(xaxis_title='Block',
                      yaxis_title=ytitle,
                      legend=legend,
                      margin=theme['graph_margins'])
    if dataframe.empty:
        return fig
    
    # Make sure we plot only 2 columns, left and right.
    columns = columns[:2]
    sides = ('negative', 'positive')
    grouped = dataframe.groupby('block')
    for name, group_df in grouped:
        for i, col in enumerate(columns):
            fig.add_trace(go.Violin(x=group_df['block'],
                                    y=group_df[col],
                                    legendgroup=col, scalegroup=col, name=col,
                                    side=sides[i],
                                    pointpos=i - 0.5,
                                    line_color=theme[col],
                                    text=[f"{col}<br />participant: {j['user']}<br />"
                                          f"block: {j['block']}<br />constraint: {j['constraint']}"
                                          for _, j in group_df.iterrows()],
                                    hoverinfo='y+text',
                                    spanmode='hard',
                                    showlegend=bool(name == dataframe['block'].unique()[0]),  # Only 1 legend.
                                    )
                          )
    
    # update characteristics shared by all traces
    fig.update_traces(meanline={'visible': True, 'color': 'dimgray'},
                      box={'visible': True, 'width': 0.5, 'line_color': 'dimgray'},
                      points='all',  # Show all points.
                      jitter=0.1,  # Add some jitter on points for better visibility.
                      scalemode='count')  # Scale violin plot area with total count.
        
    block_range = [dataframe['block'].astype(float).min() - 0.5, dataframe['block'].astype(float).max() + 0.5]
    fig.update_layout(violingap=0, violingroupgap=0, violinmode='overlay', hovermode='closest')
    fig.update_xaxes(tickvals=dataframe['block'].unique(), range=block_range)
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='LightPink')
    return fig


def generate_lines_plot(dataframe, y_var, errors=None, by='user', color_col=None):
    """ Intended for use with either df1/df2 or parallel/orthogonal.
    
    :param dataframe: Values to plot
    :type dataframe: pandas.DataFrame
    :param y_var: Y-axis variable. What is being plotted?
    :type y_var: str
    :param errors: column name for standard deviations.
    :type errors: str|None
    :param by: Line group. What any column in dataframe will be grouped by.
    :type by: str
    :param color_col: Column containing keys for colors from theme.

    :return: Figure object of graph.
    :rtype: plotly.graph_objs.Figure
    """
    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
    )
    
    hover_data = ['block', 'constraint', y_var, errors, by] if errors else ['block', 'constraint', y_var, by]
    try:
        block_range = [dataframe['block'].astype(float).min() - 0.5, dataframe['block'].astype(float).max() + 0.5]
        fig = px.line(data_frame=dataframe, x='block', y=y_var, line_group=by, error_y=errors,
                      color=color_col, color_discrete_map=theme,
                      hover_data=hover_data, range_x=block_range,
                      render_mode='webgl')
        fig.update_xaxes(tickvals=dataframe['block'].unique())
    except (KeyError, ValueError):
        fig = go.Figure()
    fig.layout.update(xaxis_title='Block',
                      yaxis_title=y_var.capitalize(),
                      legend=legend,
                      margin=theme['graph_margins'])
    return fig
