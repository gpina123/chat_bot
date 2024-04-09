import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash_mantine_components as dmc
import pandas as pd
import numpy as np

import call_chatgpt_v2


external_stylesheets = [
    'https://www.w3schools.com/w3css/4/w3.css',
    'https://fonts.googleapis.com/css?family=Lato',
    'https://fonts.googleapis.com/css?family=Montserrat',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)


df = pd.DataFrame(columns=['User Input'])

# Footer
footer = html.Footer(className='w3-container w3-black w3-padding-64 w3-center w3-opacity', children=[
    html.P(children='Energy Services, IST, 3rd Term 2023/2024', style={'display': 'inline-block', 'verticalAlign': 'middle',"bottom-margin":"0px","padding-bottom":"0px"})
])

# Header
header = html.Header(className='w3-container w3-center', style={'padding': '128px 16px', 'background-color': '#009DE0'}, children=[
    html.H1(className='w3-margin w3-jumbo', children='Energy Services AI Assistant'),
    html.P(className='w3-xlarge', children='by Diogo Lemos, Gonçalo Almeida, João Tavares, and Vasco Nunes'),
])

# Chatbot Layout
chatbot_layout = html.Div(id='link2', className='w3-row-padding w3-light-grey w3-padding-64 w3-container', children=[
    html.Div(className='w3-content', children=[
        html.Div(children=[ 
               dbc.Row([
                    dbc.Col(
                        #dcc.Input(id='input-box', type='text', placeholder='Enter your text',
                        #          style={"width":"100%"}),
                        dmc.Textarea(
                            id="input-box",
                            label="User input",
                            placeholder="Enter your text",
                            style={"width": "100%", "font-size": "80px"},
                            autosize=True,
                            minRows=1,
                        ),
                        width=9,
                        style={"display":"inline"}
                    ),
                    dbc.Col(
                        html.Button('Submit', id='button'), width=3,style={"display":"inline"}
                    ),
                ], id='input-container'),
            dbc.Row(id='output_from_gpt'),
        ])
    ]),
])

# Help Layout
help_layout = html.Div(id='link1', className='w3-row-padding w3-light-grey w3-padding-64 w3-container', children=[
    html.Div(className='w3-content', children=[
        html.Div(children=[ 
                html.H1(children='Data Visualization'),
                html.H5(className='w3-padding-32', style={'text-align': 'justify'}, children='Meteorological data plays a crucial role in forecasting energy consumption, providing valuable insights that enable more accurate predictions and efficient resource allocation. By integrating meteorological variables such as temperature, humidity, pressure, and radiance into energy consumption models, analysts can better understand the complex relationship between weather patterns and energy usage.'),
                html.P(className='w3-text', style={'text-align': 'justify'}, children='Here, you can visualize energy consumption and meteorological data from 2017 to 2019 (up to March). Simply drag the slider to select the desired year and explore the variables you wish to inspect.'),
        ])
    ]),
])

app.layout = html.Div([
    ################################################ MENU ################################################
    html.Div(className='w3-top', style={'background-color': '#009DE0'}, children=[
        html.Div(className='w3-bar w3-card w3-left-align w3-large', children=[
            html.Button(className='w3-bar-item w3-button w3-hide-medium w3-hide-large w3-right w3-padding-large w3-hover-white w3-large',
                        id='menu-button', children=[
                            html.I(className='fa fa-bars')
                        ]),
            html.A(className='w3-bar-item w3-button w3-padding-large w3-white', href='/', children='ChatBot'),
            html.A(className='w3-bar-item w3-button w3-hide-small w3-padding-large w3-hover-white', href='/help', children='Help'),
        ]),
        html.Div(id='navDemo', className='w3-bar-block w3-white w3-hide w3-hide-large w3-hide-medium w3-large', children=[
            html.A(className='w3-bar-item w3-button w3-padding-large', href='/', children='ChatBot'),
            html.A(className='w3-bar-item w3-button w3-padding-large', href='/help', children='Help'),
        ])
    ]),

    # Header
    header,

    # Content will be rendered based on the URL
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),

    footer
])

########################################## Buttons
@app.callback(
    Output('navDemo', 'className'),
    [Input('menu-button', 'n_clicks')],
    prevent_initial_call=True
)
def toggle_navbar(n_clicks):
    if n_clicks % 2 == 1:
        return 'w3-bar-block w3-white w3-show w3-large'
    else:
        return 'w3-bar-block w3-white w3-hide w3-large'

########################################

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')],
)
def display_page(pathname):
    if pathname == '/help':
        return help_layout
    else:
        return chatbot_layout

##########################################

@app.callback(
    Output('output_from_gpt', 'children'),
    [Input('button', 'n_clicks')],
    [State('input-box', 'value')],
    prevent_initial_call=True
)
def update_output(n_clicks, value):
    if n_clicks is None:
        return ''
    else:
        user_input = value
        call_chatgpt_v2.get_user_input(user_input)
        output = call_chatgpt_v2.get_gpt_output()
        function_response = output.get("function_response")
        if function_response is None:
            display = html.Div(f'{output["text_response"]}')
        elif isinstance(function_response,str):
            display = html.Div([html.Div(f'{output["text_response"]}'),
                                html.Div(f'{function_response}')])
        else:
            display = html.Div([html.Div(f'{output["text_response"]}'),
                                dcc.Graph(figure=function_response)])
        
        return html.Div([
            display,
            html.Div([
                    dmc.Textarea(
                        id="input-box",
                        label="User input",
                        placeholder="Enter your text",
                        style={"width": "100%"},
                        autosize=True,
                        minRows=1,
                    ),
                    html.Button('Submit', id='button')  # This is the correct ID
            ]),
            html.Div(id="output_from_gpt")
            ])
        
if __name__ == '__main__':
    app.run_server(debug=True)
