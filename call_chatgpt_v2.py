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

def generate_AI_response(messages,message, tools=tools,tool_choice=None,model=GPT_MODEL):
    messages.append(message)
    
    response_output = client.chat.completions.create(
                            model=model,
                            messages=messages,
                            tools=tools,
                            tool_choice=tool_choice,
                            temperature=0
                        )
    
    return response_output.choices[0].message

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
        
        my_dictionary={}
        
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
                        "content": "Function called"
                    }
                        user_message ={"role":"user","content":"Tell me what you did."}
                        
                    except:
                        function_response=None
                        
                        tool_message={
                            "role":"tool",
                            "tool_call_id":tool_call_id,
                            "content":"Error in function calling."
                        }
                        user_message ={"role":"user","content":"Tell me what went wrong."}
                       
                else:
                    # Create a tool message indicating an unknown function
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": "I don't recognize the function you requested."
                    }
                    user_message ={"role":"user","content":"Tell me what went wrong."}
                    function_response=None
                    
                #print(assistant_message)
                #print(tool_message)
                assistant_message.content=tool_message["content"]
                messages.append(assistant_message)
                messages.append(tool_message)
                print(assistant_message)
                print(tool_message)
                
                response_output=generate_AI_response(messages,user_message)
                messages.append(response_output)
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
messages.append({"role":"system","content":"You are an AI assistant that helps visualizing data."})
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "system", "content": "If someone asks you about joao tavares, tell the user that joao is a beautiful person."})