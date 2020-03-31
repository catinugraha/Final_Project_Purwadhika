import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import seaborn as sns
import numpy as np
import dash_table
from dash.dependencies import Input, Output, State

def generate_table(dataframe, page_size=10):
    return dash_table.DataTable(
        id='dataTable',
        columns=[{
        "name": i,
        'id': i
        } for i in dataframe.columns],
        data=dataframe.to_dict('records'),
        page_action='native',
        page_current=0,
        page_size=page_size
    )

year=[1896, 1900, 1904, 1906, 1908, 1912, 1920, 1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 1994, 1998, 2002, 2006, 2010, 2014]
list_ct=['total_contingent','total_medal','total_gold','total_silver','total_bronze']
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1("Hello Dash"),
    html.Div(children="""
        Dash: A web application framework for Python.
        """),
        html.Div(children=[
                html.Div([
                    html.P('Season'),
                    dcc.Dropdown(value='Summer',
                    id='filter-season',
                    options=[{'label':'Summer','value':'Summer'},
                    {'label':'Winter','value':'Winter'}])
                ],className='col-4'),
                html.Div([
                    html.P('Category'),
                    dcc.Dropdown(value='total_contingent',
                    id='filter-category',
                    options=[{'label':'Total Contingent','value':'total_contingent'},
                    {'label':'Total Medal','value':'total_medal'},
                    {'label':'Total Gold','value':'total_gold'},
                    {'label':'Total Silver','value':'total_silver'},
                    {'label':'Total Bronze','value':'total_bronze'}])
                ],className='col-4')
            ],className='row'),
            html.Br(),
            html.Div([
                dcc.Graph(id='world-map')
            ]),
            html.Br(),
            html.Div([html.P('Year'),
                dcc.Slider(
                    id='year-slider',
                    min=1896,
                    max=2016,
                    step=2,
                    value=1896,
                    marks={i: {'label':'{}'.format(i),
                    'style':{'fontSize': 14,'writing-mode': 'vertical-lr','text-orientation': 'sideways'}} for i in year})
            ])],
style={
    'maxWidth':'1200px',
    'margin': '0 auto'
})

@app.callback(
    Output(component_id='world-map',component_property='figure'),
    [Input(component_id='filter-season',component_property='value'),
    Input(component_id='filter-category',component_property='value'),
    Input(component_id='year-slider',component_property='value')]
)

def update_table(season,category,year):
    cm99 = pd.read_csv('listCM.csv')
    for i in cm99['season'].unique():
        for j in cm99['year'].unique():
            if i == season and j == year:
                list_ctr=cm99[(cm99['season']==i) & (cm99['year']==j)]['country']
                list_cat=cm99[(cm99['season']==i) & (cm99['year']==j)][category]
    fig = go.Figure(data=go.Choropleth(
    locations = list_ctr,
    locationmode='country names',
    z = list_cat,
    text = list_ctr,
    colorscale = 'Blues',
    autocolorscale=True,
    reversescale=True,
    showscale = True,
    marker_line_color='black',
    marker_line_width=0.5,
#     colorbar_tickprefix = '$',
    colorbar_title = '{}'.format(category)
    ))
    lay_out = go.Layout(
        title_text='Olympics {}'.format(year),
        geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular')) 
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)