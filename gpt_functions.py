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


def update_graph(selected_options,start_date,end_date,type_graph):
    
    start_date=pd.to_datetime(start_date)
    end_date=pd.to_datetime(end_date)
    
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
        return "Your time series graph is completed."
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
        
        return "Your boxplot graph is completed"
        
        
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
        return "Your histogram graph is completed"