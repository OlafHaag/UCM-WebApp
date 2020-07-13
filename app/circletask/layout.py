""" This module contains all the dash components visible to the user and composes them to a layout. """
from datetime import datetime
from pathlib import Path
import string

import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash_table.Format import Format, Scheme, Symbol
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np

from . import analysis

# Set pandas plotting backend to ploty. Requires plotly >= 4.8.0.
pd.options.plotting.backend = 'plotly'

app_route = 'circletask'

top_templates_path = Path(__file__).parents[1] / 'templates'
nav_html = (top_templates_path / 'nav.html').read_text()
dashapp_templates_path = Path(__file__).parent / 'templates'
intro_html = (dashapp_templates_path / 'information.html').read_text()
refs_html = (dashapp_templates_path / 'references.html').read_text()

# Index page.
html_layout = f'''<!DOCTYPE html>
                    <html>
                        <head>
                            {{%metas%}}
                            <title>{{%title%}}</title>
                            {{%favicon%}}
                            {{%css%}}
                        </head>
                        <body>
                            {nav_html}
                            {intro_html}
                            {{%app_entry%}}
                            {refs_html}
                            <footer>
                                {{%config%}}
                                {{%scripts%}}
                                {{%renderer%}}
                            </footer>
                        </body>
                    </html>'''

# ToDo: Move style setttings to less.
# Body
theme = {'font-family': 'Lobster',
         'background-color': '#e7ecf7',
         'height': '60vh',
         'graph_margins': {'l': 40, 'b': 40, 't': 40, 'r': 10},
         # Use colors consistently to quickly grasp what is what.
         'df1': 'cornflowerblue',
         'df2': 'palevioletred',
         'sum': 'peru',
         'parallel': 'lawngreen',
         'orthogonal': 'salmon',
         'colors': px.colors.qualitative.Plotly,
         }


def create_header():
    """ The header for the dashboard. """
    header_style = {'background-color': theme['background-color'], 'padding': '1.5rem', 'textAlign': 'center'}
    header = html.Header(html.H2(children="EDA Dashboard", style=header_style))
    return header


###############
# Components. #
###############
def generate_upload_component(upload_id):
    """ Component to receive new data to upload to the server.
    
    :param upload_id: Unique identifier for the component.
    :type upload_id: str
    :rtype: dash_core_components.Upload.Upload
    """
    upload_widget = dcc.Upload(id=upload_id,
                               children=html.Div(["Drag and Drop or ", html.A("Select CSV Files"), " for upload."]),
                               accept=".csv",
                               style={'width': '100%',
                                      'height': '60px',
                                      'lineHeight': '60px',
                                      'borderWidth': '1px',
                                      'borderStyle': 'dashed',
                                      'borderRadius': '5px',
                                      'textAlign': 'center'},
                               # Allow multiple files to be uploaded
                               multiple=True)
    return upload_widget


def generate_daterange_picker():
    date_picker = html.Div([
        html.Div([html.Label('Date range:')], style={'marginInline': '5px', 'display': 'inline-block'}),
        dcc.DatePickerRange(
            id='date-picker-range',
            min_date_allowed=datetime(2020, 6, 5),
            initial_visible_month=datetime(2020, 6, 5),
            display_format='MMM Do, YYYY',
            start_date=datetime(2020, 6, 5).date(),
        ),
    ], style={'display': 'inline-block', 'margin': '0 3rem'})
    return date_picker


def generate_user_select(dataframe):
    """ Dropdown to filter for specific user data. """
    if dataframe.empty:
        options = dict()
    else:
        options = [{'label': p, 'value': p} for p in dataframe['user'].unique()]
    user_select = html.Div([
        html.Div([html.Label('Participant')], style={'marginInline': '5px', 'display': 'inline-block'}),
        html.Div([dcc.Dropdown(
            id='user-IDs',
            options=options,
            value=[],
            placeholder='Filter...',
            clearable=True,
            multi=True,
        )], style={'verticalAlign': 'middle', 'display': 'inline-block', 'minWidth': '100px'}),
    ])
    return user_select


def dash_row(*children):
    """
    :param children: Components to be displayed side by side, e.g. table and figure.
    :type children: List
    :return:
    """
    row = html.Div(className='row', children=[*children])
    sep = html.Hr()
    return row, sep


###############
#   Graphs.   #
###############
def get_figure_div(graph, num, description):
    """ Wrapper for graphs to add description and consistent APA-like styling.
    
    :param graph: Graph object.
    :type graph: dash_core_components.Graph.Graph
    :param num: Ordinal number of figure.
    :type num: int
    :param description: Description of graph.
    :type description: str
    """
    fig = html.Div(className='six columns',
                   children=[html.Div([graph], className='pretty_container'),
                             dcc.Markdown(f"*Figure {num}.*  {description}")])
    return fig


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


def generate_grab_figure(dataframe, feature='duration'):
    """ Plot duration, onset or release of slider grabs.
    
    :param dataframe: Trial data.
    :type dataframe: pandas.DataFrame
    :param feature: Which variable of df1 and df2 is to be plotted. One of 'duration', 'grab', 'release'
    :type feature: str
    
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
                      yaxis_title=f"Grab {'Onset' if feature=='grab' else feature.capitalize()} (s)",
                      legend=legend,
                      margin=theme['graph_margins'])
    if dataframe.empty:
        return fig
    
    grouped = dataframe.groupby('block')
    for name, group_df in grouped:
        fig.add_trace(go.Violin(x=group_df['block'],
                                y=group_df[f'df1_{feature}'],
                                legendgroup='df1', scalegroup='df1', name='df1',
                                side='negative',
                                line_color=theme['df1'],
                                showlegend=bool(name == dataframe['block'].unique()[0]),
                                )
                      )
        fig.add_trace(go.Violin(x=group_df['block'],
                                y=group_df[f'df2_{feature}'],
                                legendgroup='df2', scalegroup='df2', name='df2',
                                side='positive',
                                line_color=theme['df2'],
                                showlegend=bool(name == dataframe['block'].unique()[0]),
                                )
                      )

    # update characteristics shared by all traces
    fig.update_traces(meanline_visible=True,
                      box_visible=True,
                      scalemode='count')  # scale violin plot area with total count
    fig.update_layout(violingap=0, violingroupgap=0, violinmode='overlay')
    fig.update_xaxes(tickvals=dataframe['block'].unique())
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='LightPink')
    return fig


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
                      xaxis={'title': 'Endpoint Value'},
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
                                    line_color=theme[col],
                                    showlegend=bool(name == dataframe['block'].unique()[0]),  # Only 1 legend.
                                    )
                          )
    
    # update characteristics shared by all traces
    fig.update_traces(meanline_visible=True,
                      box_visible=True,
                      scalemode='count')  # scale violin plot area with total count
    fig.update_layout(violingap=0, violingroupgap=0, violinmode='overlay')
    fig.update_xaxes(tickvals=dataframe['block'].unique())
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='LightPink')
    return fig


###############
#   Tables.   #
###############
def get_table_div(table, num, title, description=None):
    """ Wrapper for table to add title and description and consistent APA-like styling.

    :param table: Table object.
    :type table: dash_table.DataTable
    :param num: Ordinal number of figure.
    :type num: int
    :param title: Title of the table.
    :type title: str
    :param description: Description of graph.
    :type description: str
    """
    table = html.Div(className='six columns',
                     children=[table,
                               html.P(f"Table {num}"),
                               dcc.Markdown(f"*{title}*"),
                               dcc.Markdown(f"*Note*: {description}" if description else ""),
                               ])
    return table


def table_type(df_column):
    """ Return the type of column for a dash DataTable.
    Doesn't work most of the time and just returns 'any'.
    Note - this only works with Pandas >= 1.0.0
    """
    if isinstance(df_column.dtype, pd.DatetimeTZDtype):
        return 'datetime',
    elif (isinstance(df_column.dtype, pd.StringDtype) or
          isinstance(df_column.dtype, pd.BooleanDtype) or
          isinstance(df_column.dtype, pd.CategoricalDtype) or
          isinstance(df_column.dtype, pd.PeriodDtype)):
        return 'text'
    elif (df_column.dtype == 'int' or
          isinstance(df_column.dtype, pd.SparseDtype) or
          isinstance(df_column.dtype, pd.IntervalDtype) or
          isinstance(df_column.dtype, pd.Int8Dtype) or
          isinstance(df_column.dtype, pd.Int16Dtype) or
          isinstance(df_column.dtype, pd.Int32Dtype) or
          isinstance(df_column.dtype, pd.Int64Dtype)):
        return 'numeric'
    else:
        return 'any'
    
    
def get_columns_settings(dataframe, order=None):
    """ Get display settings of columns for tables.
    
    :param dataframe: Data
    :type dataframe: pandas.DataFrame
    :param order: Custom order for columns. Use position, in case names change.
    :type order: list[int]
    :return: List of dicts. Columns displaying float values have special formatting.
    :rtype: list
    """
    columns = list()
    if order is None:
        cols = dataframe.columns
    else:
        # Reorder columns.
        try:
            cols = [dataframe.columns[i] for i in order]
        except IndexError:
            print("WARNING: Order of columns out of range.")
            cols = dataframe.columns
    for c in cols:
        # Nicer column names. Exclude df1 and df2 from renaming.
        label = c.replace("_", " ").title().replace("Df1", "df1").replace("Df2", "df2")
        if dataframe[c].dtype == 'float':
            columns.append({'name': label,
                            'id': c,
                            'type': 'numeric',
                            'format': Format(precision=2, scheme=Scheme.fixed)})
        else:
            columns.append({'name': label, 'id': c, 'type': table_type(dataframe[c])})
    return columns


def get_pca_columns_settings(dataframe):
    """ Get display settings of columns for PC vs. UCM table.

    :param dataframe: Angles between principal components and ucm vectors.
    :type dataframe: pandas.DataFrame
    :return: List of dicts. Columns displaying float values have special formatting.
    :rtype: list
    """
    columns = list()
    for c in dataframe.columns:
        if dataframe[c].dtype == 'float':
            columns.append({'name': c,
                            'id': c,
                            'type': 'numeric',
                            'format': Format(nully='N/A',
                                             precision=2,
                                             scheme=Scheme.fixed,
                                             symbol=Symbol.yes,
                                             symbol_suffix=u'˚')})
        else:
            columns.append({'name': c, 'id': c, 'type': table_type(dataframe[c])})
    return columns


def generate_table(dataframe, table_id):
    """ Get a table to display data with conditional formatting.
    
    :param dataframe: Data to be displayed
    :param table_id: Unique identifier for the table.
    :return: Dash DataTable
    """
    table = dash_table.DataTable(
        id=table_id,
        data=dataframe.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in dataframe.columns],
        export_format='csv',
        filter_action='native',
        sort_action='native',
        sort_mode='multi',
        style_table={'height': theme['height'], 'marginBottom': '0px'},
        style_header={'fontStyle': 'italic',
                      'borderTop': '1px solid black',
                      'borderBottom': '1px solid black',
                      'textAlign': 'center'},
        style_filter={'borderBottom': '1px solid grey'},
        fixed_rows={'headers': True, 'data': 0},
        style_as_list_view=True,
        style_cell={
            'minWidth': '0px', 'maxWidth': '20px',  # 'width': '20px',
            'whiteSpace': 'normal',  # 'no-wrap',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'textAlign': 'center',
        },
        style_cell_conditional=[
            {'if': {'column_id': 'user'},
             'width': '5%'},
            {'if': {'column_id': 'session'},
             'width': '7%'},
            {'if': {'column_id': 'block'},
             'width': '5%'},
            {'if': {'column_id': 'constraint'},
             'width': '8%'},
            {'if': {'column_id': 'outlier'},
             'width': '5.5%'},
            # 'display': 'none',
        ],
        style_data={'border': '0px'},
        style_data_conditional=[
            {'if': {'filter_query': '{outlier} = 1'},
             'color': 'red'},
        ],
        css=[{
            'selector': '.dash-cell div.dash-cell-value',
            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
        }],
    )
    return table


def generate_simple_table(dataframe, table_id):
    """ Create a table just showing the data.
        No sorting or filterring.

    :param dataframe: data to be displayed.
    :param table_id: Unique identifier for the table.
    :return: DataTable
    """
    
    table = dash_table.DataTable(
        id=table_id,
        data=dataframe.to_dict('records'),
        columns=get_columns_settings(dataframe),
        export_format='csv',
        style_header={'fontStyle': 'italic',
                      'borderTop': '1px solid black',
                      'borderBottom': '1px solid black',
                      'textAlign': 'center'},
        style_cell={
            'minWidth': '0px', 'maxWidth': '20px',  # 'width': '20px',
            'whiteSpace': 'normal',  # 'no-wrap',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'textAlign': 'center',
        },
        style_data={'border': '0px', 'textAlign': 'center'},
        # Bottom header border not visible, fake it with upper border of row 0.
        style_data_conditional=[{
            "if": {"row_index": 0},
            'borderTop': '1px solid black'
        }],
        css=[{
            'selector': '.dash-cell div.dash-cell-value',
            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
        }],
    )
    return table


#######################
# Compose components. #
#######################
def create_content():
    """ Compose widgets into a layout. """
    # Start with an empty dataframe, gets populated by callbacks anyway.
    df = pd.DataFrame()
    # Create widgets.
    upload_widget = generate_upload_component('upload-data')
    filter_hint = "You can filter numerical and string columns by =, !=, <, <=, >=, >. " \
                  "You can also type parts of a string for filtering."
    # ToDo: Hint that you can hide/show items by clicking/dbl-clicking the legend.

    trials_graph = html.Div(className='six columns',
                            children=[html.Div([html.Button(id='refresh-btn',
                                                            n_clicks=0,
                                                            children='Refresh from DB',
                                                            ),
                                                generate_daterange_picker(),
                                                ], style={'marginBottom': '3rem'}),
                                      html.Div(className='pretty_container',
                                               children=[
                                                   dcc.Graph(id='scatterplot-trials', style={'height': theme['height']})
                                                         ]),
                                      html.Div(style={'display': 'flex',
                                                      'flex-wrap': 'wrap',
                                                      'justify-content': 'space-between',
                                                      'align-items': 'baseline',
                                                      'margin': '1rem auto',
                                                      },
                                               children=[generate_user_select(df),
                                                         dcc.Checklist(id='pca-checkbox',
                                                                       options=[{'label': 'Principal Components',
                                                                                 'value': 'Show'}],
                                                                       value=['Show'],
                                                                       ),
                                                         html.Span(children=[
                                                             html.Label("Contamination",
                                                                        style={'marginInline': '5px',
                                                                               'display': 'inline-block'}),
                                                             dcc.Input(id='contamination',
                                                                       type="number",
                                                                       inputMode='numeric',
                                                                       min=0.0,
                                                                       max=0.5,
                                                                       value=0.05,
                                                                       step=0.001,
                                                                       style={'width': '8rem'},
                                                                       ),
                                                                   ],
                                                                   ),
                                                         ]),
                                      dcc.Markdown("*Figure 3.* Final state values of degrees of freedom 1 and 2, "
                                                   "colored by block. "
                                                   "The subspace of task goal 1 is presented as a line. The 2 possible "
                                                   "goals for the concurrent tasks are represented as larger discs."
                                                   " Only one of these goals is selected for a constrained block. "
                                                   "Principle components are displayed as arrows. "
                                                   "A threshold for outliers calculated using the robust covariance "
                                                   "method is drawn as a thin black line. Set the contamination rate "
                                                   "to adjust the threshold.")
                                      ],
                            )
    
    trials_table = get_table_div(generate_table(df, 'trials-table'), 1,
                                 "Final state values of slider positions",
                                 "The goal of task 1 is to match the sum of df1 and df2 "
                                 "to be equal to 125. Outliers are identified using the robust "
                                 "covariance method and are colored in red.  \n" + filter_hint)
    
    grab_onset_graph = get_figure_div(dcc.Graph(id='onset-dfs'), 4, "Onset of sliders for df1 and df2 being grabbed.")
    
    grab_duration_graph = get_figure_div(dcc.Graph(id='duration-dfs'), 5,
                                         "Duration of sliders for df1 and df2 being grabbed.")
    
    hist_graph_dfs = get_figure_div(dcc.Graph(id='histogram-dfs'), 6, "Histograms of final state values for df1 and "
                                                                      "df2 compared to normal distributions.")
    
    hist_graph_sum = get_figure_div(dcc.Graph(id='histogram-sum'), 7, "Histogram of the sum of df1 and df2 "
                                                                      "compared to a normal distribution.")
    
    corr_table = get_table_div(generate_simple_table(df, 'corr-table'), 2,
                               "Pairwise Pearson correlation coefficients",
                               "There is a reciprocal suppression when: "
                               "r(sum,df1) > 0, r(sum, df2)>0, r(df1,df2)<0."
                               )
    
    # ToDo: histogram of residuals
    pca_graph = get_figure_div(dcc.Graph(id='barplot-pca'), 8, "Explained variance by different principal components"
                                                               " in percent.")
    
    pca_table = get_table_div(generate_simple_table(df, 'pca-table'), 3,
                              "Divergence between principal components "
                              "and the space parallel or orthogonal to the theoretical UCM"
                              )
    
    var_graph = get_figure_div(dcc.Graph(id='barplot-variance', style={'height': theme['height']}), 9,
                               "**A** Variance of the sum of df1 and df2 grouped by block and participant. **B** "
                               "Mean of the sum of df1 and df2.")
    var_graph.style = {'marginTop': '70px'}  # Match it to the table y position.
    
    # Table of projections' length mean and variance.
    # ToDo: When LaTeX rendering is supported in dash Markdown, convert.
    proj_table = html.Div(className='twelve columns',
                          children=[generate_table(df, 'desc-table'),
                                    html.P("Table 4"),
                                    dcc.Markdown("*Mean and variance of projection's lengths*"),
                                    # Following text contains math formulas.
                                    # Keep the math sections short, as they do not wrap when resizing.
                                    html.Span([html.I("Note: "),
                                               html.Span("The lengths are the absolute values of coefficients "
                                                         "$a$ and $b$ in $\\vec{x}-\\bar{x} = "
                                                         "a\\hat{v}_{\parallel UCM} + b\\hat{v}_{\\perp UCM}$ "
                                                         "with $\\vec{x}$ being a 2-dimensional data point "
                                                         "$\\vec{x}=[df1 \\; df2]$ and $\\bar{x}$ "
                                                         "being the mean vector. "
                                                     "$\\|\\hat{v}_{\parallel UCM}\\|=\\|\\hat{v}_{\\perp UCM}\\|=1$.",)
                                               ]),
                                    dcc.Markdown(filter_hint)
                                    ])
    
    # Tie widgets together to layout.
    content = html.Div([
        dcc.Store(id='datastore', storage_type='memory'),
        dcc.Store(id='proj-store', storage_type='memory'),
        dcc.Store(id='contour-store', storage_type='memory'),
        dcc.Store(id='pca-store', storage_type='memory'),
        
        html.Div(id='div-data-upload',
                 children=[upload_widget],
                 # Hide Div in non-debug environment.
                 style={'paddingTop': '20px', 'display': 'none'}),
        html.Div(id='output-data-upload'),
        
        # Figures and Tables.
        html.Div(style={'textAlign': 'left'},
                 children=[*dash_row(trials_graph, trials_table,
                                     html.Div([
                                         html.P(id='removal-hint', style={'display': 'inline'}),
                                         html.P(id='filtered-hint', style={'display': 'inline'})],
                                         className='six columns')),
                           *dash_row(grab_onset_graph, grab_duration_graph),
                           *dash_row(hist_graph_dfs, hist_graph_sum),
                           *dash_row(corr_table),
                           *dash_row(pca_graph, pca_table),
                           *dash_row(proj_table),
                           *dash_row(var_graph),
                           ]),
    ])
    return content


def create_footer():
    """ A footer for the dashboard. """
    def get_footer_link(label, url, url_text=None):
        if not url_text:
            url_text = url
        link = html.P(children=[html.Span(f"{label } "), html.A(url_text, href=url, target='_blank')])
        return link
        
    dash_link = get_footer_link("Built with", 'https://github.com/plotly/dash', "Plotly Dash")
    src_link = get_footer_link("Source Code on", 'https://github.com/OlafHaag/UCM-WebApp/', "GitHub")
    app_link = get_footer_link("Data acquired with", 'https://github.com/OlafHaag/NeuroPsyResearchApp', "NeuroPsy Research App")
    
    footer_style = {'background-color': theme['background-color'], 'padding': '0.5rem'}
    footer = html.Footer(children=[dash_link, src_link, app_link], style=footer_style)
    return footer


def serve_layout():
    """ Pack dash components into a Div with id dash-container. """
    layout = html.Div(
        children=[create_header(), create_content(), create_footer()],
        className='container',
        style={'font-family': theme['font-family']},
        id='dash-container'
    )
    return layout
