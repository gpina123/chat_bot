tools = [
    {########################### PLOT GRAPH
    "type": "function",
        "function":{
            "name": "plot_graph",
            "description": "Plot a graph",
            "parameters": {
                "type": "object",
                "properties": {
                    "selected_options": {
                        "type": "array",
                        "description": "The selected options for plotting the graph",
                        "items": {
                            "type":'string',
                            "enum":["Power (kW)","Temperature (C)","Humidity (%)","WindSpeed (m/s)",
                                "Pressure (mbar)","SolarRad (W/m2)","rain (mm/h)", 'Power-1', 'Power-week',
                                'Week Day', 'Month', 'Holiday', 'Holiday or Weekend', 'Power RM-2H',
                                'Power RM-4H', 'Temperature RM-2H', 'Temperature RM-4H',
                                'Solar Irradiance RM-2H', 'Solar Irradiance RM-4H', 'Power RStd-2H',
                                'Power RStd-4H', 'Temperature RStd-2H', 'Temperature RStd-4H',
                                'Solar Irradiance RStd-2H', 'Solar Irradiance RStd-4H', 'Power deriv1',
                                'Power deriv2', 'Hour', 'Hour sin', 'Hour cos',]
                        }
                    },
                    
                    "start_date": {
                        "type": "string",
                        "description": "The start date for the data range. It does not matter the format",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "The end date for the data range. It does not matter the format",
                    },
                    "type_graph": {
                        "type": "string",
                        "enum": ["time_series","boxplot","histogram"],
                        "description": "The type of graph to plot (only 'time_series', 'boxplot' and 'histogram')",
                    },
                    "building": {
                        "type": "string",
                        "description": "The selected building for plotting the graph",
                        "enum":["Central","Civil","South Tower","North Tower","IST"]
                    },
                    "num_bins": {
                        "type": "number",
                        "description": "The number of bins the user wishes when he selects the histogram type of graph",
                    },
                },
                "required": ["selected_options", "start_date", "end_date", "type_graph", "building"]
            },
        }
    },
    {########################### FEATURE SELECTION
        "type": "function",
        "function":{
            "name": "feature_selector",
            "description": "Performs feature selection - it receives as input the features and selection method and returns a graph which compares the scores of the different features",
            "parameters": {
                "type": "object",
                "properties": {

                    "building": {
                        "type": "string",
                        "description": "The selected building for plotting the graph",
                        "enum":["Central","Civil","South Tower","North Tower","IST"]
                    },
                    
                    "selected_features": {
                        "type": "array",
                        "description": "The selected options for plotting the graph",
                        "items": {
                            "type":'string',
                            "enum":['Temperature (C)',
                                    'Humidity (%)', 'WindSpeed (m/s)', 'Pressure (mbar)', 'SolarRad (W/m2)',
                                    'rain (mm/h)', 'Power-1', 'Power-week', 'Hour', 'Hour sin', 'Hour cos',
                                    'Week Day', 'Month', 'Holiday', 'Holiday or Weekend', 'Power RM-2H',
                                    'Power RM-4H', 'Temperature RM-2H', 'Temperature RM-4H',
                                    'Solar Irradiance RM-2H', 'Solar Irradiance RM-4H', 'Power RStd-2H',
                                    'Power RStd-4H', 'Temperature RStd-2H', 'Temperature RStd-4H',
                                    'Solar Irradiance RStd-2H', 'Solar Irradiance RStd-4H', 'Power deriv1',
                                    'Power deriv2']
                        }
                    },
                     "selection_method": {
                        "type": "string",
                        "description": "The selection method used",
                        "enum":["kBest-F-Value", "kBest-MI", "Forest-Regressor"]
                    },
                    "start_date": {
                        "type": "string",
                        "description": "The start date for the data range",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "The end date for the data range",
                    },
                },
                "required": ["building", "selected_features", "selection_method"],
            },
        }
    },
    {########################### CREATE FORECAST MODEL
        "type": "function",
        "function":{
            "name": "create_regression_model",
            "description": "Creates a regression/forecast model",
            "parameters": {
                "type": "object",
                "properties": {
                    "selected_features": {
                        "type": "array",
                        "description": "The selected features to build the model",
                        "items": {
                            "type": "string",
                            "enum": ['Temperature (C)',
                                'Humidity (%)', 'WindSpeed (m/s)', 'Pressure (mbar)', 'SolarRad (W/m2)',
                                'rain (mm/h)', 'Power-1', 'Power-week', 'Hour', 'Hour sin', 'Hour cos',
                                'Week Day', 'Month', 'Holiday', 'Holiday or Weekend', 'Power RM-2H',
                                'Power RM-4H', 'Temperature RM-2H', 'Temperature RM-4H',
                                'Solar Irradiance RM-2H', 'Solar Irradiance RM-4H', 'Power RStd-2H',
                                'Power RStd-4H', 'Temperature RStd-2H', 'Temperature RStd-4H',
                                'Solar Irradiance RStd-2H', 'Solar Irradiance RStd-4H', 'Power deriv1',
                                'Power deriv2']
                        }
                    },
                    "start_date": {
                        "type": "string",
                        "description": "The start date for the data range",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "The end date for the data range",
                    },
                    "type_model": {
                        "type": "string",
                        "enum": ["Linear Regression","Decision Tree","Random Forest","Gradient Boosting","Bootstrapping","Neural Networks",],
                        "description": "The type of regression model",
                    },
                    "selected_building": {
                        "type": "string",
                        "description": "The selected building",
                        "enum":["Central","Civil","South Tower","North Tower","IST"]
                    },
                },
                "required": ["selected_options", "start_date", "end_date","type_model","selected_building"],
            },
        },
    },
    {########################### APPLY FORECAST MODEL
        "type": "function",
        "function":{
            "name": "forecast",
            "description": "Uses a regression/forecast model created previously to make the forecast of energy consumption.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "The start date for the data range",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "The end date for the data range",
                    },
                },
                "required": ["start_date", "end_date"],
            },
        },
    },
    {########################### SHOW PLOTS FOR TEST DATA
        "type": "function",
        "function":{
            "name": "model_plots",
            "description": "Creates a plot that evaluates the performance of the regression/forecast model on the test data",
            "parameters": {
                "type": "object",
                "properties": {
                    "selected_plot": {
                        "type": "string",
                        "description": "The type of plot requested.",
                        "enum":['time_series','scater_plot','table']
                    },
                    "selected_metrics": {
                        "type": "array",
                        "description": "The selected metrics to appear on the table",
                        "items": {
                            "type": "string",
                            "enum": ['MAE','MBE','MSE','RMSE','cvRMSE','NMBE']
                        }
                    },
                },
                "required": ["selected_plot"],
            },
        },
    },
    {########################### SHOW PLOTS FOR THE FORECAST
        "type": "function",
        "function":{
            "name": "forecast_plots",
            "description": "After the forecast model has been aplied to same data, it creates a plot that evaluates the performance of the regression/forecast model",
            "parameters": {
                "type": "object",
                "properties": {
                    "selected_plot": {
                        "type": "string",
                        "description": "The type of plot requested.",
                        "enum":['time_series','scater_plot','table']
                    },
                    "selected_metrics": {
                        "type": "array",
                        "description": "The selected metrics to appear on the table",
                        "items": {
                            "type": "string",
                            "enum": ['MAE','MBE','MSE','RMSE','cvRMSE','NMBE']
                        }
                    },
                },
                "required": ["selected_plot"],
            },
        },
    },
]