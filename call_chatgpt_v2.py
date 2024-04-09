import os
import openai
from openai import OpenAI
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import plotly.graph_objects as go

import gpt_functions
import tools

'''
openai.api_key ='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'

GPT_MODEL = "gpt-3.5-turbo-0613"

client = OpenAI(api_key = openai.api_key)
'''

openai.api_key = os.getenv("OPENAI_API_KEY")

GPT_MODEL = "gpt-3.5-turbo-0613"

client = OpenAI()

tools = tools.tools

def get_user_input(user_input):
    message={"role": "user", "content": f"{user_input}"}
    print(user_input)
    messages.append(message)

def get_gpt_output():
    output_gpt = chat_completion_request(messages,tools=tools)
    return output_gpt

def generate_AI_response(messages, tools=tools,tool_choice="auto",model=GPT_MODEL):
    response_output = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            tools=tools,
                            tool_choice=tool_choice,
                            temperature=0
                        )
    
    return response_output.choices[0].message

#@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, tools=tools, tool_choice="auto", model=GPT_MODEL):
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
        
        my_dictionary={}
        
        # Check if the assistant's message contains tool calls
        if assistant_message.tool_calls is not None:
            for tool_call in assistant_message.tool_calls:
                #tool_call = assistant_message.tool_calls[0]
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
                        "content": f"Specify what you have done based on {str(arguments)}."
                    }
                        assistant_message.content="Function called."
                    
                    except ValueError:
                        function_response=None
                        
                        tool_message={
                            "role":"tool",
                            "tool_call_id":tool_call_id,
                            "content":"Tell the user the date range has no content."
                        }
                        assistant_message.content="Error in function calling."
                    except:
                        function_response=None
                        
                        tool_message={
                            "role":"tool",
                            "tool_call_id":tool_call_id,
                            "content":"What went wrong?"
                        }
                        assistant_message.content="Error in function calling. If there is an argument missing, warn the user."
                        
                else:
                    # Create a tool message indicating an unknown function
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": "I don't recognize the function you requested."
                    }
                    #user_message ={"role":"user","content":"Tell me what went wrong."}
                    function_response=None
                
            messages.append(assistant_message)
            messages.append(tool_message)
            print("\n\n\nASSISTANT MESSAGE")
            print(assistant_message)
            print("\n\n\nTOOL MESSAGE")
            print(tool_message)
            
            response_output=generate_AI_response(messages)
            messages.append(response_output)
            print("\n\n\nRESPONSE OUTPUT MESSAGE")
            print(response_output)
            
            my_dictionary.update({"text_response":response_output.content})
            my_dictionary.update({"function_response":function_response})
                
        else:
            messages.append(assistant_message)
            function_response=None
            my_dictionary.update({"function_response":function_response})
            
            if isinstance(assistant_message,dict):
                my_dictionary.update({"text_response":assistant_message["content"]})
            else:
                my_dictionary.update({"text_response":assistant_message.content})
                
        return my_dictionary
        
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

        
messages = []
messages.append({"role":"system","content":"You are an AI assistant that helps visualizing data. You only have access to data from 2017, 2018 and 2019."})
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "system", "content": '''When refering to the functions and options, try to be the most natural possible. Don't refer their actual names, but yes their role'''})
messages.append({"role": "system", "content": '''When the user asks who was the responsible for this work, tell him this project is part
                 of the energy services course, and is authored by Diogo Lemos, Gonçalo Almeida, João Tavares and Vasco Nunes.'''})
'''
Example prompts:
Please plot a graph of the type histogram for the Power_kW from January 9th, 2019 to December 20th, 2019.
Please plot a graph of the type histogram for the Power consumption from January 5th, 2019 to December 31st, 2019.
Please create a forecast model using temperature, pressure and solar radiance, with the data from January 1st to February 30th 2019, using random forest, for the Central building.
Use that model to make the energy consumpsion forecast for March and April 2019.
Give me the error metrics of the forecast.
Can you perfrom feature selection with temp_C,  HR and 'windSpeed_m/s as feature and kBest-MI as selection method
Can you perfrom feature selection all features possible and kBest-MI as selection method
Can you perfrom feature selection all features possible and Random forest as selection method
Can you perfrom feature selection all features possible and Random forest as selection method from 1st January 2019 to 3rd March 2019
Can you perfrom feature selection all features possible and k_best f value as selection method from 1st January 2019 to 3rd March 2019 for the Central building
Can you perfrom feature selection all features possible and rfe as selection method from 1st January 2019 to 3rd March 2019 for the Central building using the 5 best features
'''
