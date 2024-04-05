import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_selection import SelectKBest # selection method
from sklearn.feature_selection import mutual_info_regression,f_regression # score metric (f_regression)
from sklearn.model_selection import train_test_split
from sklearn import  metrics
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import  linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
import plotly.express as px

import dash_bootstrap_components as dbc

import gpt_interface

###############################################################################################################

all_data = pd.read_csv('All_data2.csv')
all_data["Date"]=pd.to_datetime(all_data["Date"], format="%Y-%m-%d %H:%M:%S")
all_data = all_data.set_index(['Building', 'Date'])

exist_model = False
exist_forecast = False

###############################################################################################################

def plot_graph(selected_options, start_date, end_date, type_graph, building, num_bins =50):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    #
    #Plot an histogram with 200 bins of the Civil building from April 2017 to April 2018 of the Power (kW) and the Temperature (C)
    print('ENTREI')
    # Ensure selected_options is a list
    if not isinstance(selected_options, list):
        selected_options = [selected_options]
    
    # Extract the data for plotting
    selected_data = all_data.xs(building).loc[start_date:end_date]
    x_data = selected_data.index
    
    # Filter data based on selected building and options
    y_data = selected_data[selected_options]

    if "time_series" in type_graph:
        # Create scatter traces for each selected option
        scatter_traces = []
        for option in selected_options:
            scatter_trace = go.Scatter(x=x_data,
                                       y=y_data[option],
                                       mode='lines',
                                       name=option)
            scatter_traces.append(scatter_trace)

        # Create layout
        layout = go.Layout(title=dict(text="Time Series Plot of the " + building + " Building", x=0.5),
                           xaxis=dict(title="Date"),
                           yaxis=dict(title="Value"))  # Assuming all selected options have the same y-axis label
        
        # Create the figure
        fig = go.Figure(data=scatter_traces, layout=layout)
        fig.write_html("time_series_plot.html")
        return fig

    elif "boxplot" in type_graph:
        # Create box plot traces for each selected option
        boxplot_traces = []
        for option in selected_options:
            boxplot_trace = go.Box(y=y_data[option], name=option)
            boxplot_traces.append(boxplot_trace)
        
        # Create layout
        layout = go.Layout(title=dict(text="Boxplot of the " + building + " Building", x=0.5),
                           yaxis=dict(title="Value"))
        
        # Create the figure
        fig = go.Figure(data=boxplot_traces, layout=layout)
        fig.write_html("boxplot_plot.html")
        return fig
        
    elif "histogram" in type_graph:
        # Create histogram traces for each selected option
        histogram_traces = []
        for option in selected_options:
            histogram_trace = go.Histogram(x=y_data[option], name=option, nbinsx=num_bins)
            histogram_traces.append(histogram_trace)
        
        # Create layout
        layout = go.Layout(title=dict(text="Histogram of the " + building + " Building", x=0.5),
                           xaxis=dict(title="Value"),
                           yaxis=dict(title="Number of occurrences"))
        
        # Create the figure
        fig = go.Figure(data=histogram_traces, layout=layout)
        fig.write_html("histogram_plot.html")
        return fig

###############################################################################################################

def feature_selector(building, selected_features,selection_method,start_date = '2019-01-10',end_date ='2019-01-30'):

    start_date=pd.to_datetime(start_date)
    end_date=pd.to_datetime(end_date)

    selected_data = all_data.xs(building).loc[start_date:end_date]
    print('feature_selector was called...')
    df_select = selected_data[selected_features]

    mask = (df_select.index >= start_date) & (df_select.index <= end_date)

    df_select = df_select[mask]

    # Can you perfrom feature selection with temp_C,  HR and 'windSpeed_m/s as feature and kBest-MI as selection method
    # Can you perfrom feature selection all features possible and kBest-MI as selection method
    # Can you perfrom feature selection all features possible and Random forest as selection method
    # Can you perfrom feature selection all features possible and Random forest as selection method from 1st January 2019 to 3rd March 2019
    # Can you perfrom feature selection all features possible and k_best f value as selection method from 1st January 2019 to 3rd March 2019 for the Central building
    # Extract the data for plotting
    Y5 = np.array(selected_data.loc[:]['Power (kW)'])
    X5 = np.array(selected_data.loc[:][selected_features])

    if selection_method == 'kBest-F-Value':
        features=SelectKBest(k=3,score_func=f_regression)
        fit=features.fit(X5,Y5) #calculates the scores using the score_function f_regression of the features
        
        x = [i for i in range(len(fit.scores_))]
        x_best = [i for i in fit.get_support(indices=True)]
        scores = fit.scores_
        columns = np.array(df_select.columns)

        
    elif selection_method == 'kBest-MI':
  
        features=SelectKBest(k=3,score_func=mutual_info_regression)
        fit=features.fit(X5,Y5)
        
        x = [i for i in range(len(fit.scores_))]
        x_best = [i for i in fit.get_support(indices=True)]
        scores = fit.scores_
        columns = np.array(df_select.columns)

        
    elif selection_method == 'Forest-Regressor':
        model = RandomForestRegressor()
        model.fit(X5, Y5)
        
        scores = model.feature_importances_
        x = [i for i in range(len(scores))]
        y_best  = np.sort(scores)[-3:]
        x_best=[]
        for i in range(len(scores)):
            for y in y_best:
                if y==scores[i]:
                    x_best.append(i)
        columns = np.array(df_select.columns)
        
    loading_output2 = None
    
    # Your plotting code here
    fig = None
    
    if selection_method:
        # Create a bar plot
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=scores, name='All Features'))
        fig.add_trace(go.Bar(x=x_best, y=scores[x_best], name='Best Features'))
    
        # Update layout
        fig.update_layout(
            title='With ' + selection_method  + ' score',
            xaxis=dict(tickvals=x, ticktext=columns, tickangle=90),
            yaxis=dict(type='log'),
            barmode='overlay',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            width = 600,
            height = 500,
        )
        
    else:
        # Display the loading spinner while the graph is being updated
        loading_output2 = dbc.Spinner(
            color="primary",
            children="Loading...",
            style={"position": "absolute", "top": "50%", "left": "50%", "transform": "translate(100%, 100%)"},
            )
        
    fig.write_html("selection.html")
    return fig

###############################################################################################################

def create_regression_model(selected_features,start_date,end_date,type_model = "Random Forest", selected_building = 'IST'):

    global exist_model
    exist_model = True
    global features
    features = selected_features
    global building
    building = selected_building
    global model
    global fig1, fig2, MAE, MBE, MSE, RMSE, cvRMSE, NMBE

    start_date=pd.to_datetime(start_date)
    end_date=pd.to_datetime(end_date)

    selected_data = all_data.xs(selected_building)
    
    # Extract the data
    Y = selected_data.loc[start_date:end_date]["Power (kW)"].values
    X = selected_data.loc[start_date:end_date][selected_features].values

    # 75% dos dados são para treino, 25% ficam para teste.
    X_train, X_test, y_train, y_test = train_test_split(X,Y)
    
    if type_model == "Linear Regression":
        print("This can take a while...")
        regr = linear_model.LinearRegression()
        regr.fit(X_train,y_train)
        y_pred = regr.predict(X_test)
        model=regr

    elif type_model == "Decision Tree":
        print("This can take a while...")
        DT_regr_model = DecisionTreeRegressor(min_samples_leaf=5)
        DT_regr_model.fit(X_train, y_train)
        y_pred = DT_regr_model.predict(X_test)
        model=DT_regr_model

    elif type_model == "Random Forest":
        print("This can take a while...")
        parameters = {'bootstrap': True,
                    'min_samples_leaf': 3,
                    'n_estimators': 200, 
                    'min_samples_split': 15,
                    'max_features': 'sqrt',
                    'max_depth': 20,
                    'max_leaf_nodes': None}
        RF_model = RandomForestRegressor(**parameters)
        RF_model.fit(X_train, y_train)
        y_pred = RF_model.predict(X_test)
        model=RF_model

    elif type_model == "Gradient Boosting":
        print("This can take a while...")
        GB_model = GradientBoostingRegressor()
        GB_model.fit(X_train, y_train)
        y_pred =GB_model.predict(X_test)
        model=GB_model

    elif type_model == "Bootstrapping":
        print("This can take a while...")
        BT_model = BaggingRegressor()
        BT_model.fit(X_train, y_train)
        y_pred =BT_model.predict(X_test)
        model=BT_model

    elif type_model == "Neural Networks":
        print("This can take a while...")
        NN_model = MLPRegressor(hidden_layer_sizes=(5,5,5))
        NN_model.fit(X_train,y_train)
        y_pred = NN_model.predict(X_test)
        model=NN_model

    fig1 = px.line(pd.DataFrame(), title='Plot of the Test data and the forecast')
    fig1.add_scatter(y=y_pred, mode='lines', name='Forecast')
    fig1.add_scatter(y=y_test, mode='lines', name='Real consumption')
    fig1.update_xaxes(title_text='')
    fig1.update_yaxes(title_text='Power (kW)')
    fig2 = px.scatter(x=y_test, y=y_pred, title='Scatter Plot of the real data and the forecast')
    fig2.update_xaxes(title_text='Real consumption')
    fig2.update_yaxes(title_text='Forecast')

    #Evaluate errors
    MAE=metrics.mean_absolute_error(y_test,y_pred) 
    MBE=np.mean(y_test- y_pred) #here we calculate MBE
    MSE=metrics.mean_squared_error(y_test,y_pred)  
    RMSE= np.sqrt(metrics.mean_squared_error(y_test,y_pred))
    cvRMSE=RMSE/np.mean(y_test)
    NMBE=MBE/np.mean(y_test)

    return fig1

##############################################################################################

def forecast(start_date,end_date):
    
    if exist_model:
        global exist_forecast
        exist_forecast = True
        global fig1_, fig2_, MAE_, MBE_, MSE_, RMSE_, cvRMSE_, NMBE_

        start_date=pd.to_datetime(start_date)
        end_date=pd.to_datetime(end_date)

        selected_data = all_data.xs(building)

        # Extract the data
        Y = selected_data.loc[start_date:end_date]["Power (kW)"].values
        X = selected_data.loc[start_date:end_date][features].values

        y_pred = model.predict(X)

        fig1_ = px.line(pd.DataFrame(), title='Time series of the Real power consumption and the Forecast')
        fig1_.add_scatter(x=selected_data.loc[start_date:end_date].index, y=Y, mode='lines', name='Real data')
        fig1_.add_scatter(x=selected_data.loc[start_date:end_date].index, y=y_pred, mode='lines', name='Forecast')
        fig1_.update_xaxes(title_text='Date')
        fig1_.update_yaxes(title_text='Power (kW)')
        fig2_ = px.scatter(x=Y, y=y_pred, title='Scatter Plot of the real data and the forecast')
        fig2_.update_xaxes(title_text='Real consumption')
        fig2_.update_yaxes(title_text='Forecast')

        #Evaluate errors
        MAE_=metrics.mean_absolute_error(Y,y_pred) 
        MBE_=np.mean(Y- y_pred) #here we calculate MBE
        MSE_=metrics.mean_squared_error(Y,y_pred)  
        RMSE_= np.sqrt(metrics.mean_squared_error(Y,y_pred))
        cvRMSE_=RMSE_/np.mean(Y)
        NMBE_=MBE_/np.mean(Y)

        return "The energy consumpsion forecast is ready!"
    
    else:
        return "Please, start by creating a model."

#############################################################################################################

def model_plots(selected_plot = 'time_series', selected_metrics = ['MAE','MBE','MSE','RMSE','cvRMSE','NMBE']):

    if exist_model:

        if selected_plot == 'time_series':
            fig1.write_html("model_time_series.html")

        elif selected_plot == 'scater_plot':
            fig2.write_html("model_scatter.html")

        elif selected_plot == 'table':
            metrics_values1 = []
            #Evaluate errors
            for metric in selected_metrics:
                if 'MAE' == metric:
                    metrics_values1.append(MAE)
                elif 'MBE' == metric:
                    metrics_values1.append(MBE)
                elif 'MSE' == metric:
                    metrics_values1.append(MSE)
                elif 'RMSE' == metric:
                    metrics_values1.append(RMSE)
                elif 'cvRMSE' == metric:
                    metrics_values1.append(cvRMSE)
                elif 'NMBE' == metric:
                    metrics_values1.append(NMBE)

            # Calculate error metrics
            error_metrics_df = pd.DataFrame({
                'Metric': selected_metrics,
                'Value': metrics_values1
            })
            # Create a table figure using graph_objects
            table_figure = go.Figure(data=[go.Table(
                header=dict(values=list(error_metrics_df.columns),
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[error_metrics_df.Metric, error_metrics_df.Value],
                        fill_color='lavender',
                        align='left'))
            ])

            table_figure.write_html("model_table.html")

        return 'The plot is ready!'
    
    else:
        return "Please, start by creating a model."

#############################################################################################################

def forecast_plots(selected_plot = 'time_series', selected_metrics = ['MAE','MBE','MSE','RMSE','cvRMSE','NMBE']):

    if exist_forecast:

        if selected_plot == 'time_series':
            fig1_.write_html("forecast_time_series.html")

        elif selected_plot == 'scater_plot':
            fig2_.write_html("forecast_scatter.html")

        elif selected_plot == 'table':
            metrics_values1 = []
            #Evaluate errors
            for metric in selected_metrics:
                if 'MAE' == metric:
                    metrics_values1.append(MAE_)
                elif 'MBE' == metric:
                    metrics_values1.append(MBE_)
                elif 'MSE' == metric:
                    metrics_values1.append(MSE_)
                elif 'RMSE' == metric:
                    metrics_values1.append(RMSE_)
                elif 'cvRMSE' == metric:
                    metrics_values1.append(cvRMSE_)
                elif 'NMBE' == metric:
                    metrics_values1.append(NMBE_)

            # Calculate error metrics
            error_metrics_df = pd.DataFrame({
                'Metric': selected_metrics,
                'Value': metrics_values1
            })
            # Create a table figure using graph_objects
            table_figure = go.Figure(data=[go.Table(
                header=dict(values=list(error_metrics_df.columns),
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[error_metrics_df.Metric, error_metrics_df.Value],
                        fill_color='lavender',
                        align='left'))
            ])

            table_figure.write_html("forecast_table.html")

        return table_figure
    
    else:
        return "Please, start by using the forecast model."