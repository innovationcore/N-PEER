from openai import OpenAI
import pandas as pd
import configparser
import re
import json

config = configparser.ConfigParser()
config.read('config.ini')
api_key = config.get('API', 'api_key')
base_url = config.get('API', 'base_url')
llm = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

metadata_filepath = 'current_metadata_official_urls.csv'

results = {}
system_prompt = """
                Please answer questions about the provided metadata file. Only use the data in this file to answer questions. 
                If a question does not pertain to the metadata file, just say so and do not attempt to answer. 
                Be sure to thoroughly check the provided metadata file before answering to ensure relevant data measures are included in your response.
                If the query matches with a certain data measure, be sure to provide the dashboard URL in your response.
                Be thorough in your response to ensure that all related or relevant measures are mentioned to the user.
                If the query doesn't directly match with just one measure, then feel free to return multiple of the most similar measures, and the
                dashboard URLs for each of those as well.
"""

prompt = 'What data measures are available for tracking drug overdose fatalities?'

with open(metadata_filepath, 'r') as file:
    file_content = file.read()
    try:
        completion = llm.chat.completions.create(
            model="DeepSeek-R1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": f"{prompt}\n"},
                    {"type": "text",
                     "text": f"[file name]: {metadata_filepath}\n[file content begin]{file_content}[file content end]"}
                ]},
            ],
            stream=True
        )
        for chunk in completion:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content is not None:
                    print(content, end="", flush=True)
    except Exception as e:
        print(f"ERROR: {e}")