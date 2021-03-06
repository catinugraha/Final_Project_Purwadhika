import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
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
        dcc.Tabs(children=[
                ## TAB-1
                dcc.Tab(value ='Tab1',label ='Data Personal Loan', children=[   
                    html.Div(children =[
                        html.Div([
                            html.P('Credit Card'),
                            dcc.Dropdown(value='None',
                            id='filter-cc',
                            options=[{'label':'None', 'value': 'None' },
                                {'label':'No', 'value': 0 },
                                {'label':'Yes', 'value': 1 }], clearable = False, className = 'col-1'),

                            html.P('Online Banking'),
                            dcc.Dropdown(value='None',
                            id='filter-online',
                            options=[{'label':'None', 'value': 'None' },
                                {'label':'No', 'value': 0 },
                                {'label':'Yes', 'value': 1 }], clearable = False, className = 'col-1'),

                            html.P('Education'),
                            dcc.Dropdown(value='None',
                            id='filter-education',
                            options=[{'label':'None', 'value': 'None' },
                                {'label':'Undergraduate', 'value': 1 },
                                {'label':'Graduate', 'value': 2 },
                                {'label':'Advance/Professional', 'value':3}], clearable = False, className = 'col-1'),

                            html.P('Personal Loan'),
                            dcc.Dropdown(value='None',
                            id='filter-loan',
                            options=[{'label':'None', 'value': 'None' },
                                {'label':'Accept', 'value': 1 },
                                {'label':'Reject', 'value': 0 }], clearable = False, className = 'col-1')
                            ], className='row-1')
                    ]),
                    html.Br(),
                    html.Div(children =[
                        html.Div([
                            html.P('Max Rows:'),
                            dcc.Input(id="filter-row",
                            placeholder="input number",
                            type="number",
                            value=10)
                        ], className='col-2'),
                    ], className='row-2'),
                    html.Br(),
                    html.Div(children =[
                        html.Div([
                        html.Button('Search', id='search-button')
                        ], className='col-3')
                    ], className='row-3'),
                    html.Br(),
                    html.Div(id='div-table',
                    children=[generate_table(data)])
                    ]),
                ## TAB-2    
                dcc.Tab(value='Tab2',label='Chart',children=[
                    html.Div(children = [
                        html.P('Category:'),
                        dcc.Dropdown(
                            id='filter-cat',
                            options=[{'label': 'Age', 'value': 'Age'},
                            {'label': 'Experience', 'value': 'Experience'},
                            {'label': 'Income per-Year ($000)', 'value': 'Income'},
                            {'label': 'Family', 'value': 'Family'},
                            {'label': 'Education', 'value': 'Education'},
                            {'label': 'Securities Account', 'value': 'Securities Account'},
                            {'label': 'CD Account', 'value': 'CD Account'},
                            {'label': 'CC Average Usage per-Month ($000)', 'value': 'CCAvg'},
                            {'label': 'Online Banking', 'value': 'Online'},
                            {'label': 'Credit Card', 'value': 'CreditCard'}],
                            value='Education',clearable = False
                        )
                    ], className = 'row col - 1'),
                    html.Br(),
                    dcc.Graph(id='graph-personalloan')]),
                ## TAB-3
                dcc.Tab(value='Tab3', label='Predict Result', children=[
                            html.Div(children=[
                                html.Div(children=[
                                    html.Div([
                                        html.P('Credit Card'),
                                        dcc.Dropdown(id='s_CreditCard',
                                        options=[{'label':'No', 'value':0},
                                                {'label':'Yes', 'value':1}],
                                        value=0, clearable = False)], className='col-3'),
                                    html.Div([
                                        html.P('Online Banking'),
                                        dcc.Dropdown(id='s_Online',
                                        options=[{'label':'No', 'value':0},
                                                {'label':'Yes', 'value':1}],
                                        value=0, clearable = False)], className='col-3'),
                                    html.Div([
                                        html.P('Education'),
                                        dcc.Dropdown(id='s_Education',
                                        options=[{'label':'Undergrad', 'value':1},
                                                {'label':'Graduate', 'value':2},
                                                {'label':'Advance/Professional', 'value':3}],
                                        value=1, clearable = False)
                                    ], className='col-3')
                                ], className = 'row'),

                                html.Br(),
                                html.Div(children=[
                                    html.Div([
                                        html.P('CC Average Usage(per-month($000))'),
                                        dcc.Input(id='s_CCAvg',
                                            type='number',
                                            value=0)
                                    ], className='col-3'),
                                    html.Div([
                                        html.P('Income($000)'),
                                        dcc.Input(id='s_Income',
                                        type='number',
                                        value=0)], 
                                        className='col-3'),
                                ], className = 'row')
                            ]),
                    html.Br(),
                    html.Div([
                        html.Button('Predict', id='buttonpredict', style=dict(width='100%'))
                    ], className='col-2 row'),       
                    html.Br(),
                    html.Div(id='display-selected-values')
                    ])
            ],
## Tabs Content Style
    content_style={
        'fontFamily': 'sans-serif',
        'borderBottom': '1px solid #d6d6d6',
        'borderLeft': '1px solid #d6d6d6',
        'borderRight': '1px solid #d6d6d6',
        'padding': '44px'
            })
    ],
    #Div Paling luar Style
    style={
        'fontFamily': 'sans-serif',
        'maxWidth': '1200px',
        'margin': '0 auto'
    })


## UPDATE TABLE CALLBACK
@app.callback(
    Output(component_id = 'div-table', component_property = 'children'),
    [Input(component_id = 'search-button', component_property = 'n_clicks')],
    [State(component_id = 'filter-row', component_property = 'value'),
    State(component_id = 'filter-cc', component_property = 'value'),
    State(component_id='filter-education',component_property='value'),
    State(component_id='filter-online',component_property='value'),
    State(component_id='filter-loan',component_property='value')])

def update_table(n_clicks, row, cc, edu, online, loan):
    data = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
    if cc != 'None':
        data = data[data['CreditCard'] == cc]
    if edu != 'None':
        data = data[data['Education'] == edu]
    if online != 'None':
        data = data[data['Online'] == online]
    if loan != 'None':
        data = data[data['Personal Loan'] == loan]
    
    children = [generate_table(data, page_size = row)]
    return children


## UPDATE GRAPH CALLBACK
@app.callback(
    Output(component_id = 'graph-personalloan', component_property='figure'),
    [Input('filter-cat', 'value')])

def update_figure(category_dropdown_name):
    df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')
    counts1 = df[df['Personal Loan']==1][category_dropdown_name].value_counts()
    counts2 = df[df['Personal Loan']==0][category_dropdown_name].value_counts()
    keys1 = counts1.index.tolist()
    keys2 = counts2.index.tolist()
    values1 = counts1.values.tolist()
    values2 = counts2.values.tolist()

    data = data=[
        go.Bar(name='Accept Personal loan', x=keys1, y=values1),
        go.Bar(name='Reject Personal Loan', x=keys2, y=values2)
    ]
    
    bar_figure = {'data': data}

    return bar_figure


## PREDICTION CALLBACK
@app.callback(
    Output('display-selected-values', 'children'),
    [Input(component_id = 'buttonpredict', component_property='n_clicks')],
    [State('s_CreditCard', 'value'),
     State('s_Online', 'value'),
     State('s_Income', 'value'),
     State('s_Education', 'value'),
     State('s_CCAvg', 'value')])

def set_display_children(n_clicks, CreditCard, Online, Income, Education, CCAvg):
    file = []
    file.append(CreditCard)
    file.append(Online)
    file.append(Income)
    file.append(Education)
    file.append(CCAvg)
    file1 = np.array(file)
    loadModel = pickle.load(open('Personal_Loan_RF_model.sav', 'rb'))
    result = loadModel.predict_proba(file1.reshape(1,5))
    hasil = []
    hasil.append("(Kemungkinan) Customer Menolak Pinjaman Personal {:.2f}%, (Kemungkinan) Customer Menerima Pinjaman Personal {:.2f}%".format(result[0,0]*100, result[0,1]*100))
    if n_clicks != None:
        return 'Hasil Prediksi:{}'.format(hasil[0])


if __name__ == '__main__':
    app.run_server(debug=True)