import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go
import pandas as pd

app_title = "Circle Task Dashboard"  # ToDo: Is there a way to get this into nav.html?
app_route = 'circletask'

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
                            <nav>
                              <a href="/"><i class="fas fa-home"></i> Home</a>
                              <a href="/{app_route}/"><i class="fas fa-chart-line"></i> {app_title}</a>
                            </nav>
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
                               children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
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
            text=[f"Participant {j}" for j in df[df['block'] == i]['user'].values],
            mode='markers',
            opacity=0.7,
            marker={'size': 15,
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
            yaxis={'title': 'Degree of Freedom 2', 'scaleanchor': "x", 'scaleratio': 1},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend=legend,
            hovermode='closest'
        )
    )
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, 100])
    # Task goal 1 visualization.
    fig.add_trace(go.Scatter(
        x=[25, 100],
        y=[100, 25],
        name="task goal 1"
    ))
    # Task goal 2 (DoF constrained) visualization.
    fig.add_scatter(y=[75, 50], x=[50, 75],
                    name="task goal 2",
                    text=["df1 contrained", "df2 constrained"],
                    mode='markers',
                    opacity=0.7,
                    marker={'size': 25})
    return fig


def generate_variance_figure(df):
    if df.empty:
        return go.Figure()

    blocks = df['block'].unique()

    fig = go.Figure(
        layout=go.Layout(
            title='Sum Variance by Block',
            xaxis={'title': 'Block'},
            yaxis={'title': 'Variance'},
            barmode='group',
            bargap=0.15,  # Gap between bars of adjacent location coordinates.
            bargroupgap=0.1,  # Gap between bars of the same location coordinate.
            margin={'l': 40, 'b': 40, 't': 40, 'r': 10},
            showlegend=False,
            hovermode='closest'
        ))
    
    grouped = df.groupby('user')
    for name, group in grouped:
        fig.add_trace(go.Bar(
            x=group['block'],
            y=group['sum var'],
            name=f'Participant {name}',
        ))
    fig.update_xaxes(tickvals=blocks)
    return fig


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
            'minWidth': '0px', 'width': '20px', 'maxWidth': '20px',
            'whiteSpace': 'no-wrap',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
        style_cell_conditional=[
            {'if': {'column_id': 'user'},
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


def create_content():
    """ Widgets. """
    # Start with an empty dataframe, gets populated by callbacks anyway.
    df = pd.DataFrame()
    # Create widgets.
    upload_widget = generate_upload_component('upload-data')
    user_chooser = generate_user_select(df)
    graph = dcc.Graph(id='scatterplot-trials')
    trials_table = generate_table(df, 'trials-table')
    var_graph = dcc.Graph(id='barplot-variance')
    var_table = generate_table(df, 'variance-table')
    
    # ToDo: table/plot for 'sum' variance.
    
    # Tie widgets together to layout.
    content = html.Div([
        dcc.Store(id='datastore', storage_type='memory'),
        dcc.Store(id='filtered-store', storage_type='memory'),
        
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
                                       graph],
                             className='six columns',
                             style={'verticalAlign': 'top'}),
                         html.Div(trials_table,
                                  className='six columns',
                                  style={'verticalAlign': 'top'})],
                     style={'textAlign': 'center'}),
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
            html.Span("Data acquired with "),
            html.A("UCMResearchApp", href="https://github.com/OlafHaag/UCMResearchApp", target="_blank"),
        ]
    )
    
    div = html.Div([p0, p1])
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
