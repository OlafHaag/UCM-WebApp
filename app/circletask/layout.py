"""
This module contains all the dash components visible to the user and composes them to a layout.
"""

from datetime import datetime
from pathlib import Path

import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash_table.Format import Format, Scheme, Symbol
import pandas as pd

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
         }


def create_header():
    """ The header for the dashboard. """
    header_style = {'background-color': theme['background-color'], 'padding': '1.5rem', 'textAlign': 'center'}
    header = html.Div(children=[html.Header(html.H2(children="EDA Dashboard", style=header_style)),
                                dcc.Markdown("To load the data and start analyzing, "
                                             "please press the **REFRESH FROM DB** button.")])
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
    div = html.Div(className='six columns',
                   children=[html.Div([graph], className='pretty_container'),
                             dcc.Markdown(f"*Figure {num}.*  {description}")])
    return div


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
        if c == 'dV':
            label = '$\\Delta V$'
        elif c == 'dVz':
            label = '$\\Delta V_z$'
        elif c == 'p-unc':
            label = 'p'
        elif c.startswith('p-'):
            label = c
        elif c == 'SS':
            label = 'Sum of Squares'
        elif c == 'MS':
            label = 'Mean Square'
        elif c == 'np2':
            label = '$\\eta_{p}^{2}$'
        elif c == 'eps':
            label = '$\\epsilon$'
        else:
            label = c.replace("_", " ").title()
        if 'Df' in label:
            label = label.replace("Df1", "df1").replace("Df2", "df2")
            
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
            {'if': {'column_id': 'condition'},
             'width': '8%'},
            {'if': {'column_id': 'block'},
             'width': '5%'},
            {'if': {'column_id': 'treatment'},
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
#        Text         #
#######################
def wilcoxon_rank_result():
    comp = html.Div(className='six columns', children=[
        html.H3("Difference between projections parallel and orthogonal to the UCM across participants."),
        html.P("A Wilcoxon signed rank test is used to compare the difference between projections parallel and "
               "orthogonal to the theoretical UCM across participants. Because of the high variability across "
               "participants a non-parametric test is used."),
        dcc.Markdown(id='wilcoxon_result', children="The result indicates that the parallel projection scores were "
                                                    "{decision}higher than the orthogonal projection scores, "
                                                    "Z={teststat}, *p = {p:.5f}*."),
    ])
    return comp


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
                                                         dcc.Checklist(id='ellipses-checkbox',
                                                                       options=[{'label': 'Ellipses',
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
    
    dof_line_plot = get_figure_div(dcc.Graph(id='df-line-plot'), 10, "Mean final state values per degree of freedom "
                                                                     "and participant. Vertical bars represent "
                                                                     "standard deviations.")

    dof_violin_plot = get_figure_div(dcc.Graph(id='df-violin-plot'), 12, "Mean final state values per degree of "
                                                                         "freedom and block across participants.")
    
    proj_line_plot = get_figure_div(dcc.Graph(id='proj-line-plot'), 11, "Projection variance per direction to UCM for "
                                                                        "each participant.")

    proj_violin_plot = get_figure_div(dcc.Graph(id='proj-violin-plot'), 13, "Variance of projections onto subspace "
                                                                            "parallel and orthogonal to UCM across "
                                                                            "participants.")
    
    corr_table = get_table_div(generate_simple_table(df, 'corr-table'), 2,
                               "Pairwise Pearson correlation coefficients",
                               "There is a reciprocal suppression when: "
                               "r(sum,df1) > 0, r(sum, df2)>0, r(df1,df2)<0."
                               )
    
    # ToDo: histogram of residuals
    pca_graph = get_figure_div(dcc.Graph(id='barplot-pca'),
                               8, "Explained variance by different principal components in percent.")
    pca_graph.style = {'marginTop': '70px'}  # Match it to the table y position next to it.
    
    pca_table = get_table_div(generate_simple_table(df, 'pca-table'), 3,
                              "Divergence between principal components "
                              "and the space parallel or orthogonal to the theoretical UCM"
                              )
    
    var_graph = get_figure_div(dcc.Graph(id='barplot-variance', style={'height': theme['height']}), 9,
                               "**A** Variance of the sum of df1 and df2 grouped by block and participant. **B** "
                               "Mean of the sum of df1 and df2.")
    
    # Table of descriptive statistics and synergy indices.
    # ToDo: When LaTeX rendering is supported in dash Markdown, convert.
    desc_table = html.Div(className='twelve columns',
                          children=[generate_table(df, 'desc-table'),
                                    html.P("Table 4"),
                                    html.P("Descriptive statistics, synergy index ($\\Delta V$) and Fisher "
                                             "z-transformed synergy index ($\\Delta V_{z}$)",
                                           style={'font-style': 'italic'}),
                                    # Following text contains math formulas.
                                    html.Span([html.I("Note: "),
                                               html.Span("The absolute averages are formed by the lengths of the "
                                                         "projections. Projections are the coefficients $a$ and $b$ in "
                                                         "$\\vec{x}-\\bar{x} = a\\hat{v}_{\parallel UCM} + "
                                                         "b\\hat{v}_{\\perp UCM}$ "
                                                         "with $\\vec{x}$ being a 2-dimensional data point "
                                                         "$\\vec{x}=[df1 \\; df2]$ and $\\bar{x}$ "
                                                         "being the mean vector. $\\hat{v}_{\parallel UCM}$ and "
                                                         "$\\hat{v}_{\perp UCM}$ are unit base vectors ("
                                                    "$\\|\\hat{v}_{\parallel UCM}\\|=\\|\\hat{v}_{\\perp UCM}\\|=1$).",)
                                               ]),
                                    dcc.Markdown(filter_hint)
                                    ])
    
    anova_table = get_table_div(generate_simple_table(df, 'anova-table'), 5,
                                "3 x (3) Two-way mixed-design ANOVA of $\\Delta V_z$ with between-factor condition "
                                "and within-factor block",
                                "If the between-subject groups are unbalanced (unequal sample sizes), "
                                "a type II sums of squares will be computed (no interaction is assumed).")
    
    posthoc_table = get_table_div(generate_simple_table(df, 'posthoc-table'), 6,
                                  "Posthoc pairwise T-tests of $\\Delta V_z$",
                                  "Only pay attention to this table if the ANOVA results merit further investigation "
                                  "(formerly known as statistical significance). The p-corr values are corrected "
                                  "p-values using the Benjamini-Hochberg False Discovery Rate method.")
    
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
                           #*dash_row(corr_table),  # Doesn't account for repeated measures
                           *dash_row(pca_graph, pca_table),
                           *dash_row(desc_table),
                           *dash_row(dof_line_plot, proj_line_plot),
                           *dash_row(dof_violin_plot, proj_violin_plot),
                           *dash_row(var_graph, wilcoxon_rank_result()),
                           *dash_row(anova_table, posthoc_table),
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
