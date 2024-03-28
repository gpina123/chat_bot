import os
import openai
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import plotly.graph_objects as go


df_raw_2019 = pd.read_csv('testData_2019_Central.csv')
df_clean_2019=df_raw_2019.copy()
df_raw_2019=df_raw_2019.rename(columns={"Central (kWh)":"Power_kW"})
df_raw_2019["Date"]=pd.to_datetime(df_raw_2019["Date"],format="%Y-%m-%d %H:%M:%S")
df_raw_2019=df_raw_2019.set_index('Date', drop=True)


#preparing data from 2019
df_clean_2019=df_clean_2019.rename(columns={"Central (kWh)":"Power_kW"})
df_clean_2019["Date"]=pd.to_datetime(df_clean_2019["Date"],format="%Y-%m-%d %H:%M:%S")
df_clean_2019=df_clean_2019.set_index('Date', drop=True)

openai.api_key = os.getenv("OPENAI_API_KEY")

GPT_MODEL = "gpt-3.5-turbo-0613"

client = OpenAI()

@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=None, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
    
def update_graph(selected_options, start_date, end_date, type_graph):
    
    # Extract the data for plotting
    x_data = df_raw_2019.loc[start_date:end_date].index
    y_data = df_raw_2019.loc[start_date:end_date][selected_options]
    
    if "time_series" in type_graph:

        #print(df_raw_2019.loc[start_date:end_date].head())
        # Create a scatter trace
        scatter_trace = go.Scatter(x=x_data,
                                    y=y_data,
                                    mode='lines',
                                    name=selected_options)

        # Create layout
        layout = go.Layout(title="Time Series Plot",
                        xaxis=dict(title="Date"),
                        yaxis=dict(title=selected_options))

        # Create the figure
        fig = go.Figure(data=[scatter_trace], layout=layout)

        fig.write_html("time_series_plot.html")
        # Show the figure
        #fig.show()
    elif "boxplot" in type_graph:
        boxplot_trace = go.Box(
            name=selected_options,
            y=y_data,
        )
        
        layout = go.Layout(title="Boxplot",
                        yaxis={"title":"Value"})
        
        fig = go.Figure(data=[boxplot_trace], layout=layout)

        fig.write_html("boxplot_plot.html")
        
        
    elif "histogram" in type_graph:
        histogram_trace=go.Histogram(
            x=y_data,
            nbinsx=50,
        )
        
        layout = go.Layout(title="Histogram",
                        xaxis={"title":"Value"},
                        yaxis={"title":"Number of occurences"})
        
        fig = go.Figure(data=[histogram_trace], layout=layout)

        fig.write_html("histogram_plot.html")
        
tools = [
    {
        "type": "function",
        "function": {
            "name": "update_graph",
            "description": "Plot a time series graph",
            "parameters": {
                "type": "object",
                "properties": {
                    "selected_options": {
                        "type": "string",
                        "description": "The selected options for plotting the graph",
                    },
                    "start_date": {
                        "type": "string",
                        "description": "The start date for the data range",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "The end date for the data range",
                    },
                    "type_graph": {
                        "type": "string",
                        "enum": ["time_series","boxplot","histogram"],
                        "description": "The type of graph to plot (only 'time_series', 'boxplot' and 'histogram')",
                    },
                },
                "required": ["selected_options", "start_date", "end_date", "type_graph"],
            },
        }
    }
]


messages = []

#"Please plot a graph of the type histogram for the Power_kW from January 5th, 2019 to December 31st, 2019."
user_input = input("Type your input:\n")
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": f"{user_input}"})
chat_response = chat_completion_request(
    messages, tools=tools
)

assistant_message = chat_response.choices[0].message
print(assistant_message)
function_call = assistant_message.tool_calls[0].function

# Parse the arguments JSON string
arguments_json = function_call.arguments
arguments_dict = json.loads(arguments_json)

# Access the individual arguments
selected_options = arguments_dict["selected_options"]
start_date = arguments_dict["start_date"]
end_date = arguments_dict["end_date"]
type_graph = arguments_dict["type_graph"]

start_date=pd.to_datetime(start_date)
end_date=pd.to_datetime(end_date)
update_graph(selected_options, start_date, end_date, type_graph)

'''
# Perform chat completion request
chat_response = chat_completion_request(messages, tools=tools)

# Iterate over the tools in the chat response
for tool in chat_response.tools:
    # Access the parameters of the function
    parameters = tool.function.parameters
    # Print the parameters
    print(parameters)

def update_graph(selected_options,start_date, end_date,type_graph):
    if "time_series" in type_graph:
        plt.plot(x=df_raw_2019.loc[start_date:end_date].index,
                 y=df_raw_2019.loc[start_date:end_date][selected_options])
        
    
    traces=[]
    if "time_series" in type_graph:
        traces.append(go.Scatter(
            x=df_raw_2019.loc[start_date:end_date].index,
            y=df_raw_2019.loc[start_date:end_date][selected_options],
            mode='lines',
        ))
        layout1 = go.Layout(yaxis={"title":"Value"})
    elif "boxplot" in type_graph:
        traces.append(go.Box(
            name=selected_options,
            y=df_raw_2019.loc[start_date:end_date][selected_options],
        ))
        layout1 = go.Layout(yaxis={"title":"Value"})
    elif "histogram" in type_graph:
        traces.append(go.Histogram(
            x=df_raw_2019.loc[start_date:end_date][selected_options],
            nbinsx=50,
        ))
        layout1 = go.Layout(xaxis={"title":"Value"},yaxis={"title":"Number of occurences"})
    

    if selected_options=="Power_kW":
        layout = go.Layout(title="Power Consumption from the Central Building (kW)")
    elif selected_options== "temp_C":
        layout = go.Layout(title='Temperature (ºC)',
                       #xaxis={'title': 'X Axis'},
                       #yaxis={'title': 'T'}
                       )
    elif selected_options=="HR":
        layout = go.Layout(title="Relative Humidity")
    elif selected_options=="windSpeed_m/s":
        layout = go.Layout(title="Wind speed (m/s)")
    elif selected_options=="windGust_m/s":
        layout = go.Layout(title="Wind gust (m/s)")
    elif selected_options=="pres_mbar":
        layout = go.Layout(title="Pressure (mbar)")
    elif selected_options=="solarRad_W/m2":
        layout = go.Layout(title="Solar Irradiation (W/m2)")
    elif selected_options=="rain_mm/h":
        layout = go.Layout(title="Rain (mm/h)")
    else:
        layout = go.Layout(title="Rain day")

    layout.update(layout1)
        
    return {'data': traces, 'layout': layout}'''