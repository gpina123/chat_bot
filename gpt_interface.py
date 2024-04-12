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
        html.Div([
            html.Div(id='output_from_gpt',style={"padding-bottom":"10px"}),
            html.Div(children=[ 
                    html.Div(
                        dmc.Textarea(
                            id="input-box",
                            placeholder="Enter your text",
                            style={"width": "100%", "font-size": "80px"},
                            autosize=True,
                            minRows=1,
                        ),
                    ),
                    html.Div([
                        html.Div(html.Div(style={'padding': '5px'})),
                        html.Button('Submit', id='button'),
                        dbc.Button(
                                    [dbc.Spinner(size="sm"), " Loading..."],
                                    color="primary",
                                    disabled=True,
                                    id="loading_button",
                                    style={"display":"None"})
                                
                    ]),
                ], id='input-container'
            ),
        ])
    ]),
])

# Help Layout
help_layout = html.Div(children=[
        html.Div(className="w3-row-padding w3-padding-64 w3-container",
                 children=[
                   html.H1(style={"margin-left":"20px","margin-right":"20px"}, children="Dear User"),
                   html.H5(style={"margin-left":"20px","margin-right":"20px"}, children="I am an AI assistant designed to help you visualize and analyze the energy consumption data at Instituto Superior Técnico, as well as to create forecasting models of energy consumption using machine learning techniques. On this page I give you a summary of the things I can do, nevertheless, feel free to ask me for help at any time!"),
                   html.P(style={"text-align":"right","margin-right":"20px"},children="- Energy Services AI Assistant")
                 ]
            
        ),
        html.Div(className='w3-row-padding w3-light-grey w3-padding-64 w3-container',
                children=[ 
                        html.H1(style={"margin-left":"20px","margin-right":"20px"},children='Data Visualization'),
                        html.P(style={"margin-left":"20px","margin-right":"20px"},children="As part of the Exploratory Data Analysis, you can select various plot types, such as time series, histogram, or box plot, to visualize all the features present in the dataset. Additionally, you have the flexibility to choose and plot as many features as you desire. In the case of the histogram, you may even choose the number of bins. For further analysis, you can request me to generate a table displaying the statistics of the selected features."),
                        html.P(style={"margin-left":"20px","margin-right":"20px"},children="Remember to specify the date range for your analysis. For additional information, simply ask me what inputs are required for me to perform the task."),
                        html.B(html.P(style={"margin-left":"20px","margin-right":"20px"},children="Prompt examples:")),
                        html.Ol(style={"margin-left":"20px","margin-right":"20px"},children=[
                            html.Li("Make a statistics table of the South Tower building in 2018 of the Power and Temperature"),
                            html.Li("Make a time series graph of IST between May 2018 and January 2019 of Power, Humidity, and Pressure"),
                            html.Li("Make a histogram with 200 bins of Central building in 2017 of Power and Temperature")
                        ])
                ]),
        html.Div(className='w3-row-padding w3-padding-64 w3-container',
                 children=[
                    html.H1(style={"margin-left":"20px","margin-right":"20px"},children="Feature Selection"),
                    html.P(style={"margin-left":"20px","margin-right":"20px"},children="With respect to feature selection, I can help you analyze and select the most relevant features for your data. To know which features are available just ask me! I can provide you with different feature selection methods, such as:"),
                    html.Ul(style={"margin-left":"20px","margin-right":"20px"},children=[
                        html.Li("kBest-F-Value: This method selects the top k features based on the F-value between each feature and the target variable."),
                        html.Li("kBest-MI: This method selects the top k features based on the mutual information between each feature and the target variable."),
                        html.Li("Forest-Regressor: This method uses a forest regressor to estimate the importance of each feature and selects the most important ones."),
                        html.Li("RFE (Recursive Feature Elimination): This method recursively eliminates features based on their importance until the desired number of features is reached.")
                    ]),
                    html.P(style={"margin-left":"20px","margin-right":"20px"},children="Once the feature selection is performed, I can also provide you with a graph that compares the scores of the different features, helping you to understand their importance. Let me know if you would like to proceed with feature selection and if you have any specific preferences or requirements!"),
                    html.B(html.P(style={"margin-left":"20px","margin-right":"20px"},children="Prompt examples:")),
                    html.Ol(style={"margin-left":"20px","margin-right":"20px"},children=[
                        html.Li("Can you perform feature selection with temperature, relative humidity, and power-1 as features and kBest-MI as the selection method."),
                        html.Li("Can you perform feature selection all features possible and kBest-F-value as the selection method from 1st January 2019 to 3rd March 2019 for the Central building displaying the 5 best features.")
                    ])
                ]),
        html.Div(className='w3-row-padding w3-light-grey w3-padding-64 w3-container',
                    children=[
                        html.H1(style={"margin-left":"20px","margin-right":"20px"},children="Forecast of the Power Consumption"),
                        html.P(style={"margin-left":"20px","margin-right":"20px"},children="To create a forecast model, I need some information from you. Please provide me with the following details:"),
                        html.Ol(style={"margin-left":"20px","margin-right":"20px"},children=[
                            html.Li("The selected features to build the model. You can choose between the features mentioned in the EDA."),
                            html.Li("The start date for the data range."),
                            html.Li("The end date for the data range."),
                            html.Li("The type of regression model you want to use. You can choose from the following options: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, Bootstrapping, or Neural Networks."),
                            html.Li("The selected building for the forecast model. You can choose from the following options: Central, Civil, South Tower, North Tower, and IST."),
                        ]),
                        html.P(style={"margin-left":"20px","margin-right":"20px"},children="Please provide me with the above information, and I will create the forecast model for you."),
                        html.P(style={"margin-left":"20px","margin-right":"20px"},children="I will take the data corresponding to the range you indicated and divide it into training (75%) and test data (25%). You can ask for plots to evaluate the performance of the model on the test data. Ask for a time series plot, a scatter plot of the forecast and real energy consumption, or a table with the main error metrics ('MAE', 'MBE', 'MSE', 'RMSE', 'cvRMSE', 'NMBE')."),
                        html.P(style={"margin-left":"20px","margin-right":"20px"},children="With a model created, you can ask me to do the energy consumption forecast for that building for any time between 2017 and April 2019."),
                        html.P(style={"margin-left":"20px","margin-right":"20px"},children="Then, you can evaluate the performance of the forecast by asking for the same plots as before (a time series, a scatter plot, or a table)."),
                        html.B(html.P(style={"margin-left":"20px","margin-right":"20px"},children="Prompt examples:")),
                        html.Ol(style={"margin-left":"20px","margin-right":"20px"},children=[
                            html.Li("Please create a forecast model using temperature, pressure and solar radiance, with the data from January 1st to February 30th 2019, using random forest, for the Central building."),
                            html.Li("Use that model to make the energy consumpsion forecast for March and April 2019."),
                            html.Li("Give me the error metrics of the forecast."),
                            html.Li("Show me the time series plot of the forecast."),
                            html.Li("Show me the scatter plot of the real consumption and the forecast.")
                        ])
                    ]   
                )
    ]),

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
    [Output('output_from_gpt', 'children'),
     Output("loading_button","style"),
     Output("input-box","value"),],
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
            display = html.Div([html.B("[Output]"),html.Div(f'{output["text_response"]}')])
        elif isinstance(function_response,str):
            display = html.Div([html.B("[Output]"), html.Div(f'{output["text_response"]}')])
        else:
            display = html.Div([html.Div([html.B("[Output]"),html.Div(f'{output["text_response"]}')]),
                                dcc.Graph(figure=function_response)])
        
        return html.Div([
                html.Div([html.B("[Input]"),html.Div(f"{value}")]),
                display,
                html.Div(id="output_from_gpt",style={"padding-bottom":"10px"})
                ],style={"padding-top":"10px"}),{"display":"None"},""
        
@app.callback(Output("loading_button","style",allow_duplicate=True),
              Input("button", "n_clicks"),
              prevent_initial_call=True)
def appear_loading_button(n_clicks):
    if n_clicks:
        return {}       
        
if __name__ == '__main__':
    app.run_server(debug=True)
