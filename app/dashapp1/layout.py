import dash_core_components as dcc
import dash_html_components as html
import dash_table

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
                                      'textAlign': 'center',
                                      'margin': '10px'},
                               # Allow multiple files to be uploaded
                               multiple=True)
    
    default_data = get_data()
    table = html.Div([
        html.H2("Trials"),
        html.H3("Across participants and blocks."),
        dash_table.DataTable(
            data=default_data.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in default_data.columns]
        ),
        html.Hr(),  # horizontal line
    ])
    
    content = html.Div([
        dcc.Store(id='datastore', storage_type='memory'),
        html.Div(id='div-data-upload',
                 children=[upload_widget],
                 # Hide Div in non-debug environment.
                 style={'display': 'none'}),
        html.Div(id='output-data-upload'),
        html.Div(id='output-data-db',
                 children=[table]),
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
