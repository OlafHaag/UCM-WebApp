import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go

from .analysis import get_data

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


def get_figure(df, users_selected=None):
    if users_selected is not None:
        dff = df.query('`participant ID` in @users_selected')
    else:
        dff = df
        
    fig = go.Figure(
        data=[
            go.Scatter(
                x=dff[dff['block'] == i]['df1'],
                y=dff[dff['block'] == i]['df2'],
                text=[f"Participant {j}" for j in dff[dff['block'] == i]['participant ID'].values],
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=f"Block {i} {dff[dff['block'] == i]['constraint'].unique()[0]}",
            ) for i in dff['block'].unique()
        ],
        layout=go.Layout(
            xaxis={'title': 'Degree of Freedom 1'},
            yaxis={'title': 'Degree of Freedom 2', 'scaleanchor': "x", 'scaleratio': 1},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            legend={'x': 0, 'y': 0},
            hovermode='closest'
        )
    )
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, 100])
    fig.add_trace(go.Scatter(
        x=[0, 125],
        y=[125, 0],
        name="task goal 1"
    ))
    fig.add_scatter(y=[75, 50], x=[50, 75],
                    name="task goal 2",
                    text=["df1 contrained", "df2 constrained"],
                    mode='markers',
                    marker={'size': 25})
    return fig


def create_content():
    """ Widgets. """
    upload_widget = dcc.Upload(id='upload-data',
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
    
    df = get_data()
    
    user_chooser = html.Div([
                        html.Div([html.Label('Participant')], style={'marginInline': '5px', 'display': 'inline-block'}),
                        html.Div([dcc.Dropdown(
                            id='user-IDs',
                            options=[{'label': p, 'value': p} for p in df['participant ID'].unique()],
                            value=[],
                            clearable=True,
                            multi=True,
                        )], style={'width': '50%', 'verticalAlign': 'middle', 'display': 'inline-block'}),
                   ], style={'marginTop': '1.5rem', })
    
    fig = get_figure(df)
    
    graph = dcc.Graph(
        id='scatterplot-trials',
        figure=fig
    )
    
    table = html.Div([
        dash_table.DataTable(
            id='trials-table',
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
        ),
    ])

    # ToDo: table/plot for 'sum' variance.
    
    content = html.Div([
        dcc.Store(id='datastore', storage_type='memory'),
        html.Div(id='div-data-upload',
                 children=[upload_widget],
                 # Hide Div in non-debug environment.
                 style={'paddingTop': '20px', 'display': 'none'}),
        html.Div(id='output-data-upload'),
        html.Div([
            html.H2("Degrees of Freedom Endpoint Variance"),
            html.H3("Across participants and blocks."),
            html.Div(id='output-data-db',
                     children=[user_chooser, graph], style={'width': '49%',
                                                            'verticalAlign': 'top',
                                                            'display': 'inline-block'}),
            html.Div([table], style={'width': '49%', 'verticalAlign': 'top', 'display': 'inline-block'}),
            html.Hr(),  # horizontal line
        ], style={'textAlign': 'center'})  # FixMe: side-by-side
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
