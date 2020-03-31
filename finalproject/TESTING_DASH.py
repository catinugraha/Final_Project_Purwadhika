import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import seaborn as sns
from dash.dependencies import Input, Output, State
import dash_table
import pickle
import numpy as np

def generate_table(dataframe, page_size=10):
    return dash_table.DataTable(
        id='dataTable',
        columns=[{
            "name": i,
            "id": i
        } for i in dataframe.columns],
        data=dataframe.to_dict('records'),
        page_action="native",
        page_current=0,
        page_size=page_size,
    )

data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

loadModel = pickle.load(open('Personal_Loan_RF_model.sav', 'rb'))

app.layout = html.Div(
    children=[
        html.H1('Personal Loan Dashboard'),
        html.Div(children='''by: Cati Nugraha Bilanguna'''),
        dcc.Tabs(
            children=[
                dcc.Tab(value='Tab1',
                    label='Data Frame',
                    children=[
                        html.Div(children=[
                            html.Div([
                                html.P('Credit Card'),
                                dcc.Dropdown(
                                    value='',
                                    id='filter_cc',
                                    options=[
                                        {'label': 'Yes', 'value': 1 },
                                        {'label': 'No', 'value': 0 }]),
                                    ], className='col-3'),

                            html.Div([
                                html.P('Education'),
                                dcc.Dropdown(
                                    value='',
                                    id='filter_edu',
                                    options=[
                                        {'label': 'Undergrad', 'value': 1 },
                                        {'label': 'Graduate', 'value': 2 },
                                        {'label': 'Advance/Professional', 'value': 3 }])
                                    ], className='col-3'),

                            html.Div([
                                html.P('Online Banking'),
                                dcc.Dropdown(
                                    value='',
                                    id='filter_ol',
                                    options=[
                                        {'label': 'Yes', 'value': 1 },
                                        {'label': 'No', 'value': 0 }])
                                    ], className='col-3'),

                            html.Div([
                                html.P('Personal Loan'),
                                dcc.Dropdown(
                                    value='',
                                    id='filter_loan',
                                    options=[
                                        {'label': 'Accept', 'value': 1 },
                                        {'label': 'Not Accept', 'value': 0 }])
                                    ], className='col-3')],
                                     className = 'row'),
                        html.Br(),
                        html.Div([
                            html.P('Max Rows:'),
                            dcc.Input(id ='filter-row',
                                        type = 'number', 
                                        value = 10)
                        ], className = 'row col-3'),

                        html.Div(children =[
                                html.Button('search', id = 'filter')
                            ],className = 'row col-4'),
                            
                        html.Div(id='div-table',
                                    children=[generate_table(data)])
                            ]
                        ),
                dcc.Tab(value='Tab2',
                    label='Scatter chart',
                    children=[
                        html.Div(children=dcc.Graph(
                            id='graph-scatter',
                            figure={
                                'data': [
                                    go.Scatter(x=data[data['Family'] == i]['Income'],
                                               y=data[data['Family'] == i]
                                               ['Income'],
                                               mode='markers',
                                               name='Family Member {}'.format(i))
                                    for i in data['Family'].unique()
                                ],
                                'layout':
                                    go.Layout(
                                        xaxis={'title': 'Family'},
                                        yaxis={'title': ' Income'},
                                        title='Tips Dash Scatter Visualization',
                                        hovermode='closest')
                            }))
                    ]),
                dcc.Tab(value='Tab3',
                        label='Predict Result',
                        children=[
                            html.Div(children=[
                                html.Div([
                                    html.P('Credit Card'),
                                    dcc.Dropdown(id='s_CreditCard',
                                    options=[{'label':'No', 'value':0},
                                            {'label':'Yes', 'value':1}],
                                    value=0)
                                ],className='row col-1'),
                                html.Div([
                                    html.P('Online Banking'),
                                    dcc.Dropdown(id='s_Online',
                                    options=[{'label':'No', 'value':0},
                                            {'label':'Yes', 'value':1}],
                                    value=0)
                                ],className='row col-2'),
                                html.Div([
                                    html.P('Income($000)'),
                                    dcc.Input(id='s_Income',
                                    type='number',
                                    value='')
                                ],className='row col-2'),
                                html.Div([
                                    html.P('Education'),
                                    dcc.Dropdown(id='s_Education',
                                    options=[{'label':'Undergrad', 'value':1},
                                            {'label':'Graduate', 'value':2},
                                            {'label':'Advance/Professional', 'value':3}],
                                    value=1)
                                ],className='row col-3'),
                                html.Div([
                                    html.P('CC Average Usage(per-month($000))'),
                                    dcc.Input(id='s_CCAvg',
                                    type='number',
                                    value='')
                                ],className='row col-4')
                            ]),
                    html.Br(),
                    html.Div(id='display-selected-values')
                        ])
            ],
            ## Tabs Content Style
            content_style={
                'fontFamily': 'Arial',
                'borderBottom': '1px solid #d6d6d6',
                'borderLeft': '1px solid #d6d6d6',
                'borderRight': '1px solid #d6d6d6',
                'padding': '44px'
            })
    ],
    #Div Paling luar Style
    style={
        'maxWidth': '1200px',
        'margin': '0 auto'
    })

@app.callback(
    Output(component_id = 'div-table', component_property = 'children'),
    [Input(component_id = 'filter', component_property = 'n_clicks')],
    [State(component_id = 'filter-row', component_property = 'value'),
    State(component_id = 'filter_cc', component_property = 'value'),
    State(component_id='filter_edu',component_property='value'),
    State(component_id='filter_ol',component_property='value'),
    State(component_id='filter_loan',component_property='value')]
)

def update_table(n_clicks,row,CreditCard,Online,Education,Personal_Loan):
    data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
    if CreditCard != '':
        data = data[data['CreditCard'] == CreditCard]
    if Online != '':
        data = data[data['Online'] == Online]
    if Education != '':
        data = data[data['Education'] == Education]
    if Personal_Loan != '':
        data = data[data['Personal Loan'] == Personal_Loan]
    children = [generate_table(data, page_size = row)]
    return children


@app.callback(
    Output('display-selected-values', 'children'),
    [Input('s_CreditCard', 'value'),
     Input('s_Online', 'value'),
     Input('s_Income', 'value'),
     Input('s_Education', 'value'),
     Input('s_CCAvg', 'value')])

def set_display_children(CreditCard, Online, Income, Education, CCAvg):
    file = []
    file.append(CreditCard)
    file.append(Online)
    file.append(Income)
    file.append(Education)
    file.append(CCAvg)
    file1 = np.array(file)
    loadModel = pickle.load(open('Personal_Loan_RF_model.sav', 'rb'))
    result = loadModel.predict(file1.reshape(1,5))
    hasil = []
    for i in result:
        if i == 1:
            hasil.append('(Kemungkinan) Customer Mau Menerima Pinjaman Personal')
        else:
            hasil.append('(Kemungkinan) Customer Menolak Pinjaman Personal')
    return 'Hasil Prediksi adalah {}'.format(hasil[0])

if __name__ == '__main__':
    app.run_server(debug=True)