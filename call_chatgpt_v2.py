import os
import openai
from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import plotly.graph_objects as go

import gpt_functions

openai.api_key = os.getenv("OPENAI_API_KEY")

GPT_MODEL = "gpt-3.5-turbo-0613"

client = OpenAI()

def parse_function_response(message):
    print("before_parse_function")
    function_call = message["function_call"]
    function_name = function_call["name"]

    print("parse_function")
    print("GPT: Called function " + function_name )

    try:
        arguments = json.loads(function_call["arguments"])

        if hasattr(gpt_functions, function_name):
            function_response = getattr(gpt_functions, function_name)(**arguments)
        else:
            function_response = "ERROR: Called unknown function"
    except:
        function_response = "ERROR: Invalid arguments"

    return (function_name, function_response)

tools = [
    {
        "type": "function",
        "function":{
            "name": "update_graph",
            "description": "Plot a graph",
            "parameters": {
                "type": "object",
                "properties": {
                    "selected_options": {
                        "type": "string",
                        "description": "The selected options for plotting the graph",
                        "enum":["Power_kW"]
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


#@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=tools,tool_choice=None, model=GPT_MODEL):
    print("hello aas")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice
        )
        
        print("hello")
        message = response.choices[0].message
        messages.append(message)
        
        if message.tool_calls:
            function_call = response.choices[0].message.tool_calls[0].function
            function_name=function_call.name        

            arguments_json = function_call.arguments
            arguments_dict = json.loads(arguments_json)
            print(arguments_dict)
            
            try:
                arguments = json.loads(function_call.arguments)
                

                if hasattr(gpt_functions, function_name):
                    function=getattr(gpt_functions, function_name)
                    function_response=function(**arguments)
                    '''
                    message = {
                    "role": "assistant",
                    "content": f"write something about {function_response}"
                    }'''
                    message = {
                        "role": "function",
                        "name": function_name,
                        "content": function_response
                    }
                else:
                    message = {
                    "role": "assistant",
                    "content": "tell the user that you don't recognize which function should you call"
                    }
                    
            except:
                message = {
                    "role": "assistant",
                    "content": "tell the user that the arguments are invalid"
                    }  
                print("ERROR: Invalid arguments")
                
        else:
            user_message = input("GPT: " + message["content"] + "\nYou: ")
            message = {
                "role": "user",
                "content": user_message
            }
            
        messages.append(message)
        
        print("before user_message")
        user_message = input("GPT: " + message["content"] + "\nYou: ")
        message = {
            "role": "user",
            "content": user_message
        }
        print("after_user_message")
        messages.append(message)
        
        print("before requesting again")
        
        chat_completion_request(messages=messages)
    
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
        
        '''   
        assistant_message = chat_response.choices[0].message
        print(assistant_message)
        function_call = assistant_message.tool_calls[0].function

        # Parse the arguments JSON string
        arguments_json = function_call.arguments
        arguments_dict = json.loads(arguments_json)
        '''
        '''
        if "function_call" in function:
            print("oafkslksks")
            function_name, function_response = parse_function_response(message)

            message = {
                "role": "function",
                "name": function_name,
                "content": function_response
            }
        else:
            user_message = input("GPT: " + message["content"] + "\nYou: ")
            message = {
                "role": "user",
                "content": user_message
            }

        chat_completion_request(message, messages)
        '''
        
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
    
        
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
#"Please plot a graph of the type histogram for the Power_kW from January 9th, 2019 to December 20th, 2019."
#user_message="Please plot a graph of the type histogram for the Power consumption from January 5th, 2019 to December 31st, 2019."
#user_input = input("GPT: Type your input:\n")
message={"role": "user", "content": "Please plot a graph of the type histogram for the Power consumption from January 5th, 2019 to December 31st, 2019."}
#message ={"role": "user", "content": f"{user_input}"}

messages.append(message)

chat_completion_request(messages,tools=tools)

'''
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
'''
# Perform chat completion request
chat_response = chat_completion_request(messages, tools=tools)

# Iterate over the tools in the chat response
for tool in chat_response.tools:
    # Access the parameters of the function
    parameters = tool.function.parameters
    # Print the parameters
    print(parameters)
'''
