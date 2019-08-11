from pathlib import Path

import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash_table.Format import Format, Scheme, Symbol
import plotly.graph_objs as go
import pandas as pd

from .analysis import get_mean_x_by, get_pca_vectors

app_title = "Circle Task Dashboard"  # ToDo: Is there a way to get this into nav.html?
app_route = 'circletask'

top_templates_path = Path(__file__).parents[1] / 'templates'
nav_html = (top_templates_path / 'nav.html').read_text()
dashapp_templates_path = Path(__file__).parent / 'templates'
intro_html = (dashapp_templates_path / 'information.html').read_text()

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
                            <footer>
                                {{%config%}}
                                {{%scripts%}}
                                {{%renderer%}}
                            </footer>
                        </body>
                    </html>'''

# Body
theme = {"font-family": "Lobster", "background-color": "#e0e0e0"}  # ToDo: get dash page theme from static css.


def create_header():
    header_style = {"background-color": theme["background-color"], "padding": "1.5rem"}
    header = html.Header(html.H1(children=app_title, style=header_style))
    return header


def generate_upload_component(upload_id):
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
    ], style={'marginTop': '1.5rem', 'textAlign': 'left'})
    return user_select


def generate_trials_figure(df):
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
            title=go.layout.Title(
                text='Trial Endpoints',
                xref="paper",
                xanchor='center',
                x=0.5,
            ),
            xaxis={'title': 'Degree of Freedom 1'},
            yaxis={'title': 'Degree of Freedom 2', 'scaleanchor': "x", 'scaleratio': 1},
            margin={'l': 40, 'b': 40, 't': 40, 'r': 10},
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
    fig.add_scatter(y=[75, 50], x=[50, 75],
                    name="task goal 2",
                    text=["task goal 2 (df1=50)", "task goal 2 (df2=50)"],
                    hoverinfo='text',
                    mode='markers',
                    opacity=0.7,
                    marker={'size': 25})
    
    fig.update_xaxes(hoverformat=".2f")
    fig.update_yaxes(hoverformat=".2f")

    return fig


def get_columns_settings(dataframe):
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
    table = dash_table.DataTable(
        id=table_id,
        data=dataframe.to_dict('records'),
        columns=[{'name': i, 'id': i} for i in dataframe.columns],
        export_format='csv',
        filter_action="native",
        sort_action="native",
        sort_mode="multi",
        style_table={'overflowX': 'scroll'},
        fixed_rows={'headers': True, 'data': 0},
        style_cell={
            'minWidth': '0px', 'maxWidth': '20px',  # 'width': '20px',
            'whiteSpace': 'normal',  # 'no-wrap',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        style_cell_conditional=[
            {'if': {'column_id': 'user'},
             'width': '10%'},
            {'if': {'column_id': 'block'},
             'width': '10%'},
            {'if': {'column_id': 'constraint'},
             'width': '10%'},
            {'if': {'column_id': 'df1',
                    'filter_query': '{df1} < 15'},
             'backgroundColor': 'red',
             'color': 'white'},
            {'if': {'column_id': 'df2',
                    'filter_query': '{df2} < 15'},
             'backgroundColor': 'red',
             'color': 'white'},
            {'if': {'column_id': 'sum',
                    'filter_query': '{sum} > 150'},
             'backgroundColor': 'red',
             'color': 'white'},
            {'if': {'column_id': 'sum var',
                    'filter_query': '{sum var} > 150'},
             'backgroundColor': 'red',
             'color': 'white'},
        ],
        css=[{
            'selector': '.dash-cell div.dash-cell-value',
            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
        }],
    )
    return table


def get_pca_annotations(pca_dataframe):
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
    """ Plot explained variance by PCs as Bar plot with cumulative explained variance. """
    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
    )
    
    layout = dict(
        title='Explained variance by different principal components',
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


def generate_pca_table(dataframe):
    dataframe.insert(0, 'pc', dataframe.index+1)
    
    table = dash_table.DataTable(
        data=dataframe.to_dict('records'),
        columns=get_pca_columns_settings(dataframe),
        export_format='csv',
        style_table={'overflowX': 'scroll'},
        fixed_rows={'headers': True, 'data': 0},
        style_cell={
            'minWidth': '0px', 'maxWidth': '20px',  # 'width': '20px',
            'whiteSpace': 'normal',  # 'no-wrap',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        css=[{
            'selector': '.dash-cell div.dash-cell-value',
            'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
        }],
    )
    return table


def generate_variance_figure(dataframe):
    if dataframe.empty:
        return go.Figure()

    blocks = dataframe['block'].unique()

    legend = go.layout.Legend(
        xanchor='right',
        yanchor='top',
        orientation='v',
        itemsizing='constant',
    )
    
    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(
                text="Sum Variance by Block",
                xref="paper",
                xanchor='center',
                x=0.5,
            ),
            xaxis={'title': 'Block'},
            yaxis={'title': 'Variance'},
            barmode='group',
            bargap=0.15,  # Gap between bars of adjacent location coordinates.
            bargroupgap=0.1,  # Gap between bars of the same location coordinate.
            margin={'l': 40, 'b': 40, 't': 40, 'r': 10},
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
            textposition="top center",
            mode='lines',
        ))
    fig.update_yaxes(hoverformat=".2f")
    
    return fig
    

def create_content():
    """ Widgets. """
    # Start with an empty dataframe, gets populated by callbacks anyway.
    df = pd.DataFrame()
    # Create widgets.
    upload_widget = generate_upload_component('upload-data')
    user_chooser = generate_user_select(df)
    graph = dcc.Graph(id='scatterplot-trials')
    trials_table = generate_table(df, 'trials-table')
    pca_graph = dcc.Graph(id='barplot-pca')
    var_graph = dcc.Graph(id='barplot-variance')
    var_table = generate_table(df, 'variance-table')
    
    # Tie widgets together to layout.
    content = html.Div([
        dcc.Store(id='datastore', storage_type='memory'),
        dcc.Store(id='pca-store', storage_type='memory'),
        
        html.Div(id='div-data-upload',
                 children=[upload_widget],
                 # Hide Div in non-debug environment.
                 style={'paddingTop': '20px', 'display': 'none'}),
        html.Div(id='output-data-upload'),
        
        html.Div([
            html.H2("Degrees of Freedom Endpoint Variance", style={'textAlign': 'center', 'marginTop': '5rem'}),
            # html.H3("Across participants and blocks."),
            html.Div(className='row',
                     children=[
                         html.Div(
                             children=[html.Div(className='row',
                                                children=[html.Button(id='refresh-btn',
                                                                      n_clicks=0,
                                                                      children='Refresh from DB'),
                                                          user_chooser]),
                                       graph,
                                       dcc.Checklist(
                                           id='pca-checkbox',
                                           options=[
                                               {'label': 'Show Principal Components', 'value': 'Show'},
                                           ],
                                           value=['Show'])
                                       ],
                             className='six columns',
                             style={'verticalAlign': 'top'}),
                         html.Div(trials_table,
                                  className='six columns',
                                  style={'verticalAlign': 'top'})],
                     style={'textAlign': 'center'}),
            html.Hr(),  # horizontal line
            html.Div(className='row',
                     children=[
                         html.Div(pca_graph,
                                  className='six columns',
                                  style={'verticalAlign': 'top'}),
                         html.Div([
                             html.H3("Divergence between principal components and UCM parallel/orthogonal space",
                                     style={'textAlign': 'center'}),
                             html.Div(id='pca-table-container',
                                      className='six columns',
                                      style={'verticalAlign': 'top'}),
                         ]),
                         ]),
            html.Hr(),  # horizontal line
            html.Div(className='row',
                     children=[
                         html.Div(var_graph,
                                  className='six columns',
                                  style={'verticalAlign': 'top', 'marginTop': '70px'}),
                         html.Div(var_table,
                                  className='six columns',
                                  style={'verticalAlign': 'top'}),
                         ]),
        ]),
        html.Hr(),  # horizontal line
    ])
    
    return content


def create_footer():
    footer_style = {"background-color": theme["background-color"], "padding": "0.5rem"}
    p0 = html.P(
        children=[
            html.Span("Built with "),
            html.A(
                "Plotly Dash", href="https://github.com/plotly/dash", target="_blank"
            ),
        ]
    )
    p1 = html.P(
        children=[
            html.Span("Soure Code on "),
            html.A("Github", href="https://github.com/OlafHaag/UCM-WebApp/", target="_blank"),
        ]
    )
    p2 = html.P(
        children=[
            html.Span("Data acquired with "),
            html.A("UCMResearchApp", href="https://github.com/OlafHaag/UCMResearchApp", target="_blank"),
        ]
    )
    
    div = html.Div([p0, p1, p2])
    footer = html.Footer(children=div, style=footer_style)
    return footer


def serve_layout():
    layout = html.Div(
        children=[create_header(), create_content(), create_footer()],
        className="container",
        style={"font-family": theme["font-family"]},
        id='dash-container'
    )
    return layout
