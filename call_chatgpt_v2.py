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

tools = [
    {
        "type": "function",
        "function":{
            "name": "plot_graph",
            "description": "Plot a graph",
            "parameters": {
                "type": "object",
                "properties": {
                    "selected_options": {
                        "type": "string",
                        "description": "The selected options for plotting the graph",
                        "enum":["Power_kW","temp_C","HR","windSpeed_m/s","windGust_m/s",
                                "pres_mbar","solarRad_W/m2","rain_mm/h","rain_day"]
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
def chat_completion_request(messages, tools=tools, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=0
        )
        
        # Get the assistant's message from the response
        assistant_message = response.choices[0].message
        
        #print(assistant_message)
        # Append the assistant's message to the list of messages
        messages.append(assistant_message)
        
        # Check if the assistant's message contains tool calls
        if assistant_message.tool_calls is not None:
            # Iterate over each tool call made by the assistant
            for tool_call in assistant_message.tool_calls:
                # Process each tool call and generate the appropriate tool message
                tool_call_id = tool_call.id
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Check if the function exists in gpt_functions
                if hasattr(gpt_functions, function_name):
                    function = getattr(gpt_functions, function_name)
                    
                    try:
                        function_response = function(**arguments)
                        
                        tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": function_response
                    }
                    except:
                        tool_message={
                            "role":"tool",
                            "tool_call_id":tool_call_id,
                            "content":"Invalid arguments!"
                        }
                else:
                    # Create a tool message indicating an unknown function
                    tool_message = {
                        "role": "assistant",
                        "tool_call_id": tool_call_id,
                        "content": "I don't recognize the function you requested."
                    }
                
                #Append the tool message to the list of messages
                messages.append(tool_message)
                #print(tool_message)
        else:
            tool_message = assistant_message
            messages.append(tool_message)
        
        # Prompt the user for input
        if isinstance(tool_message,dict):
            user_message = input("GPT: " + tool_message["content"] + "\nYou: ")
        else:
            if tool_message.content is not None:
                user_message = input("GPT: " + tool_message.content + "\nYou: ")
            else:
                user_message = input("GPT: " + "content none" + "\nYou: ")
        # Create a user message with the input
        user_message = {
            "role": "user",
            "content": user_message
        }
        
        # Append the user message to the list of messages
        messages.append(user_message)
        
        # Recursively call chat_completion_request with updated messages
        chat_completion_request(messages, tools=tools, tool_choice=tool_choice, model=model)
    
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

        
messages = []
messages.append({"role":"system","content":"You are an AI assistant that helps visualizing data."})
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
#"Please plot a graph of the type histogram for the Power_kW from January 9th, 2019 to December 20th, 2019."
#"Please plot a graph of the type histogram for the Power consumption from January 5th, 2019 to December 31st, 2019."
user_input = input("GPT: Type your input."+"\nYou: ")
message={"role": "user", "content": f"{user_input}"}
#message ={"role": "user", "content": f"{user_input}"}

messages.append(message)

chat_completion_request(messages,tools=tools)
