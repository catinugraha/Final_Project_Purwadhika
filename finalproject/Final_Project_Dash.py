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

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

loadModel = pickle.load(open('Personal_Loan_RF_model.sav', 'rb'))

app.layout = html.Div(children = [
    html.Div(children=[
                html.Div([
                    html.P('Credit Card'),
                    dcc.Dropdown(id='s_CreditCard',
                    options=[{'label':'No', 'value':0},
                            {'label':'Yes', 'value':1}],
                    style={'height': '2px',
                            'width': '100px',
                            'font-size': "100%",
                            'min-height': '1px'},
                    value=0)
                ],className='col-1')
            ]),
    html.Br(),
    html.Div(children=[
                html.Div([
                    html.P('Online Banking'),
                    dcc.Dropdown(id='s_Online',
                    options=[{'label':'No', 'value':0},
                            {'label':'Yes', 'value':1}],
                    style={'height': '2px',
                            'width': '100px',
                            'font-size': "100%",
                            'min-height': '1px'},
                    value=0)
                ],className='col-2')
            ]),
    html.Br(),
        html.Div(children=[
                html.Div([
                    html.P('Income($000)'),
                    dcc.Input(id='s_Income',
                    type='number',
                    value='')
                ],className='col-3')
            ]),
    html.Br(),
        html.Div(children=[
                html.Div([
                    html.P('Education'),
                    dcc.Dropdown(id='s_Education',
                    options=[{'label':'Undergrad', 'value':1},
                            {'label':'Graduate', 'value':2},
                            {'label':'Advance/Professional', 'value':3}],
                    style={'height': '2px',
                            'width': '150px',
                            'font-size': "100%",
                            'min-height': '1px'},
                    value=1)
                ],className='col-4')
            ]),
    html.Br(),
        html.Div(children=[
                html.Div([
                    html.P('CC Average Usage(per-month($000))'),
                    dcc.Input(id='s_CCAvg',
                    type='number',
                    value='')
                ],className='col-3')
            ]),
    html.Br(),
    html.Div(id='display-selected-values')
])

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