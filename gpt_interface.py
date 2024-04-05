import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import dash_mantine_components as dmc
import pandas as pd
import numpy as np

import call_chatgpt_v2

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL, dbc.themes.BOOTSTRAP, 'styles.css'], suppress_callback_exceptions=True)

df = pd.DataFrame(columns=['User Input'])

app.layout = dbc.Container([
    dbc.Row(dcc.Markdown("# Energy Services AI Assistant")),
    dbc.Row([dcc.Markdown("### Authors:"),html.P("Diogo Lemos"),html.P("Gonçalo Almeida"),html.P("João Tavares"),html.P("Vasco Nunes")]),
    dbc.Row([
        dbc.Col(
            #dcc.Input(id='input-box', type='text', placeholder='Enter your text',
            #          style={"width":"100%"}),
            dmc.Textarea(
                id="input-box",
                label="User input",
                placeholder="Enter your text",
                style={"width": "100%"},
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
    #dbc.Row(id="historic")
    #html.Div(id='output-container'),
])

@app.callback(
    Output('output_from_gpt', 'children'),
     #Output('historic', 'children')],
     #Output('output-container', 'children')
    [Input('button', 'n_clicks')],
    [State('input-box', 'value')],
    prevent_initial_call=True
)
def update_output(n_clicks, value):
    if n_clicks is None:
        return '', ''
    else:
        user_input = value
        
        '''global df
        new_row = pd.DataFrame({'User Input': user_input})
        pd.concat([df,new_row])
        #df = df.concat({'User Input': user_input}, ignore_index=True)
        table = html.Div([
            html.H3('User Input history'),
            dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)
        ])'''
        # Call the function to get user input
        call_chatgpt_v2.get_user_input(user_input)
        
        # Call the function to get GPT output
        output = call_chatgpt_v2.get_gpt_output()
        function_response = output.get("function_response")
        if function_response is None:
            # Convert the figure to a dictionary
            display = html.Div(f'{output["text_response"]}')
        else:
            display = html.Div([html.Div(f'{output["text_response"]}'),
                                dcc.Graph(figure=function_response)])
        
        
        # Create the Dash Graph component
        
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
                    html.Button('Submit', id='button')
            ]),
            html.Div(id="output_from_gpt")
            ])
        
if __name__ == '__main__':
    app.run_server(debug=True)
