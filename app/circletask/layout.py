""" This module contains all the dash components visible to the user and composes them to a layout. """
from pathlib import Path

import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash_table.Format import Format, Scheme, Symbol
import plotly.graph_objs as go
import plotly.figure_factory as ff
import pandas as pd
import numpy as np

from .analysis import get_mean_x_by, get_pca_vectors

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
            text=[f"df1={j['df1']}<br />df2={j['df2']}<br />Sum={j['sum']:.2f}<br />Participant {j['user']}"
                  for _, j in df[df['block'] == i].iterrows()],
            hoverinfo='text',
            mode='markers',
            opacity=0.7,
            marker={'size': 10,
                    'line': {'width': 0.5, 'color': 'white'}
                    },
            name=f"Block {i} {'|'.join(df[df['block'] == i]['constraint'].unique())}",
        ) for i in df['block'].unique()]

    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
    )
    
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            xaxis={'title': 'Degree of Freedom 1'},
            yaxis={'title': 'Degree of Freedom 2', 'scaleanchor': 'x', 'scaleratio': 1},
            margin=theme['graph_margins'],
            legend=legend,
            hovermode='closest',
        )
    )
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, 100])
    # Task goal 1 visualization.
    fig.add_trace(go.Scatter(
        x=[25, 100],
        y=[100, 25],
        mode='lines',
        name="task goal 1"
    ))
    # Task goal 2 (DoF constrained) visualization.
    fig.add_scatter(y=[62.5], x=[62.5],
                    name="task goal 2",
                    text=["task goal 2 (df1=df2)"],
                    hoverinfo='text',
                    mode='markers',
                    opacity=0.7,
                    marker={'size': 25})

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


def generate_histograms(dataframe):
    """ Plot distribution of data to visually check for normal distribution.
    
    :param dataframe: Data of df1, df2 and their sum.
    :type dataframe: pandas.DataFrame
    """
    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
    )
    
    cols = list(dataframe.columns)
    cols.reverse()
    if dataframe.empty:
        fig = go.Figure()
        fig.update_xaxes(range=[0, 100])
    else:
        # Columns we want to plot histograms for. Display order is reversed.
        data = [dataframe[c] for c in cols]
        # Create distplot with curve_type set to 'normal'.
        fig = ff.create_distplot(data, cols, curve_type='normal')  # Override default 'kde'.

    fig.layout.update(legend=legend,
                      yaxis={'title': 'Probability Density'},
                      xaxis={'title': 'Endpoint Value'},
                      margin=theme['graph_margins'],
                      )
    return fig


def get_columns_settings(dataframe):
    """ Get display settings of columns for tables.
    
    :param dataframe: Data
    :type dataframe: pandas.DataFrame
    :return: List of dicts.  Columns displaying float values have special formatting.
    :rtype: list
    """
    columns = list()
    for c in dataframe.columns:
        if dataframe[c].dtype == 'float':
            columns.append({'name': c,
                            'id': c,
                            'type': 'numeric',
                            'format': Format(precision=2, scheme=Scheme.fixed)})
        else:
            columns.append({'name': c, 'id': c})
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
             'width': '10%'},
            {'if': {'column_id': 'block'},
             'width': '10%'},
            {'if': {'column_id': 'constraint'},
             'width': '10%'},
            #{'if': {'column_id': 'outlier'},
            # 'display': 'none'}
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


def get_pca_annotations(pca_dataframe):
    """ Generate display properties of principal components for graphs.
    
    :param pca_dataframe: Results of PCA.
    :type pca_dataframe: pandas.DataFrame
    :return: List of properties for displaying arrows.
    :rtype: list
    """
    # Visualize the principal components as vectors over the input data.
    arrows = list()
    # ToDo: groupby
    vectors = [get_pca_vectors(pca_dataframe)]
    if vectors:
        for group in vectors:
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
                arrowcolor='#636363'
            )
                for v0, v1 in group])
    return arrows


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
    )
    
    layout = dict(
        yaxis={'title': 'Explained variance in percent'},
        margin={'l': 60, 'b': 40, 't': 40, 'r': 10},
    )
    
    if dataframe.empty:
        data = []
    else:
        trace1 = dict(
            type='bar',
            x=[f'PC {i+1}' for i in dataframe.index],
            y=dataframe['var_expl'],
            name='Individual'
        )
        """ Since we didn't reduce dimensionality, cumulative explained variance will always add up to 100%.
        # Add cumulative visualization.
        cum_var_exp = dataframe['var_expl'].cumsum()
        trace2 = dict(
            type='scatter',
            x=[f'PC {i+1}' for i in dataframe.index],
            y=cum_var_exp,
            name='Cumulative'
        )
        layout.update(legend=legend,
                      annotations=list([
                          dict(
                              x=1.02,
                              y=1.05,
                              xref='paper',
                              yref='paper',
                              text='Explained Variance',
                              showarrow=False,)]),
                      )
        data = [trace1, trace2]
        """
        data = [trace1]
    
    fig = dict(data=data, layout=layout)
    return fig


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
            columns.append({'name': c, 'id': c})
    return columns


def generate_variance_figure(dataframe):
    """ Barplot of variance in the sum of df1 and df2 per block.
    
    :param dataframe: Data of variances.
    :type dataframe: pandas.DataFrame
    """
    if dataframe.empty:
        fig = go.Figure()
        fig.layout.update(yaxis={'title': 'Variance'})
        return fig

    blocks = dataframe['block'].unique()

    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
        itemsizing='constant',
    )
    
    fig = go.Figure(
        layout=go.Layout(
            xaxis={'title': 'Block'},
            yaxis={'title': 'Variance'},
            barmode='group',
            bargap=0.15,  # Gap between bars of adjacent location coordinates.
            bargroupgap=0.1,  # Gap between bars of the same location coordinate.
            margin=theme['graph_margins'],
            showlegend=True,
            legend=legend,
            annotations=list([
                dict(x=1.02,
                     y=1.05,
                     xref='paper',
                     yref='paper',
                     text='Mean Variance',
                     showarrow=False, )]),
            hovermode='closest'
        ))
    
    grouped = dataframe.groupby('user')
    for name, group in grouped:
        fig.add_trace(go.Bar(
            x=group['block'],
            y=group['sum var'],
            name=f'Participant {name}',
            showlegend=False,
        ))
    fig.update_xaxes(tickvals=blocks)

    # Add mean across participants by block
    mean_vars = get_mean_x_by(dataframe, 'sum var', by='block')
    for block, v in mean_vars.iteritems():
        fig.add_trace(go.Scatter(
            x=[block - 0.5, block, block + 0.5],
            y=[v, v, v],
            name=f"Block {block}",
            hovertext=f'Block {block}',
            hoverinfo='y',
            textposition='top center',
            mode='lines',
        ))
    fig.update_yaxes(hoverformat='.2f')
    
    return fig


#######################
#
#######################
def get_table_div(table, num, title, description=None):
    table = html.Div(className='six columns',
                     children=[table,
                               html.P(f"Table {num}"),
                               dcc.Markdown(f"*{title}*"),
                               dcc.Markdown(f"*Note*: {description}" if description else ""),
                               ])
    return table


def get_figure_div(graph, num, description):
    fig = html.Div(className='six columns',
                   children=[html.Div([graph], className='pretty_container'),
                             dcc.Markdown(f"*Figure {num}.*  {description}")])
    return fig


def dash_row(*children):
    """
    :param children: Components to be displayed side by side, e.g. table and figure.
    :type children: List
    :return:
    """
    row = html.Div(className='row', children=[*children])
    sep = html.Hr()
    return row, sep


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
                            children=[html.Button(id='refresh-btn',
                                                  n_clicks=0,
                                                  children='Refresh from DB',
                                                  style={'marginBottom': '3rem'}),
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
                                                                       step=0.01,
                                                                       style={'width': '7rem'},
                                                                       ),
                                                                   ],
                                                                   ),
                                                         ]),
                                      dcc.Markdown("*Figure 3.* Endpoint values of degrees of freedom 1 and 2, colored "
                                                   "by block. "
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
                                 "Endpoint values of slider positions",
                                 "The goal of task 1 is to match the sum of df1 and df2 "
                                 "to be equal to 125. Outliers are identified using the robust "
                                 "covariance method and are colored in red.  \n" + filter_hint)
    
    hist_graph_dfs = get_figure_div(dcc.Graph(id='histogram-dfs'), 4, "Histograms of endpoint values for df1 and df2 "
                                                                      "compared to normal distributions.")
    
    hist_graph_sum = get_figure_div(dcc.Graph(id='histogram-sum'), 5, "Histogram of the sum of df1 and df2 "
                                                                      "compared to a normal distribution.")
    # ToDo: histogram of residuals
    pca_graph = get_figure_div(dcc.Graph(id='barplot-pca'), 6, "Explained variance by different principal components"
                                                               " in percent.")
    
    pca_table = get_table_div(generate_simple_table(df, 'pca-table'), 2,
                              "Divergence between principal components "
                              "and the space parallel or orthogonal to the theoretical UCM"
                              )
    
    var_graph = get_figure_div(dcc.Graph(id='barplot-variance'), 7, "Variance of the sum of df1 and df2 "
                                                                    "grouped by block and participant.")
    var_graph.style = {'marginTop': '70px'}  # Match it to the table y position.
    
    var_table = get_table_div(generate_table(df, 'variance-table'), 3,
                              "Means and variances of df1, df2 and their sum",
                              filter_hint)
    
    # Table of projections' length mean and variance.
    # ToDo: When LaTeX rendering is supported in dash Markdown, convert.
    proj_table = html.Div(className='six columns',
                          children=[generate_simple_table(df, 'proj-table'),
                                    html.P("Table 4"),
                                    dcc.Markdown("*Mean and variance of projection's lengths*"),
                                    # Following text contains math formulas.
                                    # Keep the math sections short, as they do not wrap when resizing.
                                    html.Span([html.I("Note: "),
                                               html.Span("The lengths are the absolute values of coefficients "),
                                               html.Span(["a \mtext{ and } b \mtext{ in }"], className='tex'),
                                               html.Span(["x-\overline{x} = aV_{\parallel UCM} + bV_{\\bot UCM}"],
                                                         className='tex'),
                                               html.Span(["\mtext{ with } x "], className='tex'),
                                               html.Span([" being a data point and "]),
                                               html.Span(["\overline{x}"], className='tex'),
                                               html.Span([" being the mean. Vectors "]),
                                               html.Span(["V_{\parallel UCM} \mtext{ and } V_{\\bot UCM}"],
                                                         className='tex'),
                                               html.Span(" are both of length 1."),
                                               ]),
                                    ])
    
    # Tie widgets together to layout.
    content = html.Div([
        dcc.Store(id='datastore', storage_type='memory'),
        dcc.Store(id='contour-store', storage_type='memory'),
        dcc.Store(id='pca-store', storage_type='memory'),
        
        html.Div(id='div-data-upload',
                 children=[upload_widget],
                 # Hide Div in non-debug environment.
                 style={'paddingTop': '20px', 'display': 'none'}),
        html.Div(id='output-data-upload'),
        
        # Figures and Tables.
        html.Div(style={'textAlign': 'left'},
                 children=[*dash_row(trials_graph, trials_table),
                           *dash_row(hist_graph_dfs, hist_graph_sum),
                           *dash_row(pca_graph, pca_table),
                           *dash_row(var_graph, var_table),
                           *dash_row(proj_table),
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
    app_link = get_footer_link("Data acquired with", 'https://github.com/OlafHaag/UCMResearchApp', "UCMResearchApp")
    
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
