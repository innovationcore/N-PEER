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

def generate_prompts(input_filepath, output_filepath):
    system_prompt = """
                    Your job is to generate prompts/questions that will be used for evaluation of an LLM.
                    The goal of that LLM is to answer questions about various data sources related to opioid overdoses, using a metadata
                    file that contains data about what each data measure is and what source it comes from. The responses from this LLM
                    will be used to help the user to determine which data measures they should look at if they are interested in certain topics.
                    I will provide you with a topic, and you must generate three prompts that can be used to evaluate the effectiveness
                    of this LLM. Try to make the prompts as varied as possible to ensure thorough evaluation of this LLM.
                    For thorough evaluation of the LLM, be sure to include noisy/unclear prompts too (and do not include any disambiguation notes).
                    Examples of potential types of prompts could be thorough like "What measure should I look at if I am interested in
                    non-fatal hospitalizations involving heroin?" or may be short and simple, like "EMS data". Please provide
                    these prompts in a consistent output format through JSON, of format {'prompt_1':'...', 'prompt_2':'...', 'prompt_3':'...'}.
                    Ensure that your output can be properly parsed into a JSON object and is enclosed in json``` ``` tags.
    """

    json_objects = []
    with open(input_filepath, 'r') as file:
        for line in file:
            topic = line.strip()
            completion = llm.chat.completions.create(
                model="DeepSeek-R1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"The topic is: {topic}\n"}
                    ]},
                ],
                stream=True
            )
            all_content = ''
            for chunk in completion:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        all_content += content
            json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", all_content)
            if json_match:
                json_string = json_match.group(1)
                try:
                    data = json.loads(json_string)
                    data['topic'] = topic
                    json_objects.append(data)
                except json.JSONDecodeError as e:
                    json_objects.append({'topic':topic, 'error':e})
            else:
                print("No JSON object found enclosed in ```json ... ```")
                json_objects.append({'topic':topic, 'error':'No JSON output provided'})
    with open(output_filepath, 'w') as f:
        json.dump(json_objects, f, indent=4)

def filter_prompts(input_filepath, output_filepath):
    system_prompt = """
                        The data provided to you is a JSON file containing topics, and three prompts for each topic.
                        Your job is to filter these down to leave just one prompt for each topic.
                        The purpose of these prompts is to evaluate the effectiveness of an LLM in answering questions about
                        data sources related to opioid overdoses, using a metadata file about each data measure.
                        The filtered prompts should be varied and effective in their evaluation of the capabilities of the LLM.
                        Be sure to include both clear/thorough prompts and short/unclear prompts in the final output.
                        Please provide the output in the same format as the input, but with only one prompt for each topic.
                        Ensure it is JSON parseable and enclosed in json``` ``` tags.
        """

    with open(input_filepath, 'r') as file:
        file_content = file.read()
        completion = llm.chat.completions.create(
            model="DeepSeek-R1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text",
                     "text": "Here is the input JSON file.\n"},
                    {"type": "text",
                     "text": f"[file name]: {input_filepath}\n[file content begin]{file_content}[file content end]"}
                ]},
            ],
            stream=True
        )
        all_content = ''
        for chunk in completion:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content is not None:
                    all_content += content
        json_match = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", all_content)
        if json_match:
            json_string = json_match.group(1)
            try:
                data = json.loads(json_string)
                with open(output_filepath, 'w') as f:
                    json.dump(data, f, indent=4)
            except json.JSONDecodeError as e:
                print(e)
        else:
            print("No JSON object found enclosed in ```json ... ```")

def format_prompt_response(prompt, response):
    """Formats a single prompt-response pair."""
    return f"PROMPT:\n{prompt}\n\nRESPONSE:\n{response}\n\n{'='*40}\n"

def format_prompt_evaluation(prompt, response):
    """Formats a single prompt-response pair."""
    return f"PROMPT:\n{prompt}\n\nEVALUATION:\n{response}\n\n{'='*40}\n"

def test_prompts(input_filepath, output_filepath, metadata_json):
    with open(input_filepath, 'r') as file:
        data = json.load(file)
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

    # with open(metadata_filepath, 'r') as file:
    #     file_content = file.read()
    for prompt_dict in data:
        prompt = prompt_dict['prompt']
        print(prompt)
        try:
            completion = llm.chat.completions.create(
                model="DeepSeek-R1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text",
                         "text": f"{prompt}\n"},
                        {"type": "text",
                         "text": f"[file name]: metadata.json\n[file content begin]{metadata_json}[file content end]"}
                    ]},
                ],
                stream=True
            )
            all_content = ''
            for chunk in completion:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        all_content += content
            match = re.search(r"</think>(.*)", all_content, re.DOTALL)
            if match:
                response = match.group(1).strip()
                results[prompt] = response
            else:
                results[prompt] = 'No response found'
        except Exception as e:
            results[prompt] = e
    with open(output_filepath, 'w', encoding='utf-8') as f:
        for prompt, response in results.items():
            formatted_entry = format_prompt_response(prompt, response)
            f.write(formatted_entry)

def read_prompt_response_pairs(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            while True:
                prompt_line = f.readline()
                if not prompt_line:  # End of file
                    break
                if prompt_line.startswith("PROMPT:"):
                    prompt = ""
                    while True:
                        line = f.readline()
                        if line.strip() == "RESPONSE:":
                            break
                        prompt += line
                    prompt = prompt.strip()

                    response = ""
                    separator_found = False
                    while True:
                        line = f.readline()
                        if not line:  # End of file within a response
                            break
                        if line.strip() == "========================================":
                            separator_found = True
                            break
                        response += line
                    response = response.strip()

                    if prompt and response:
                        yield (prompt, response)
                    elif prompt:
                        # Handle case where response might be missing before separator/EOF
                        yield (prompt, "")
                    elif response:
                        # Handle case where prompt might be missing
                        yield ("", response)

                    if not separator_found and not not line: # If no separator and not EOF after response
                        # Attempt to find the next PROMPT to continue the loop
                        while True:
                            next_line = f.readline()
                            if not next_line:
                                break
                            if next_line.startswith("PROMPT:"):
                                # We'll process this in the next iteration
                                # Need to "unread" this line somehow, or just let the loop handle it
                                break
                # Skip lines that don't start with "PROMPT:"
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return

def evaluate_prompts(input_filepath, output_filepath, metadata_json):
    results = []
    system_prompt = """
                    Your job is to evaluate a prompt/response pair to determine if the response is adequate.
                    The prompts concern data measures stored in a metadata file provided to you, and the answers were LLM-generated.
                    When evaluating these responses, use the metadata file for help and specifically answer these four questions:
                    1. Is the question appropriately answered in a relevant and understandable way without misinterpretation?
                    2. Does the response provide relevant measures/dashboards that addresses all parts of the question without ignoring any important available measures/dashboards?
                    3. Does the response acknowledge a lack of appropriate data when applicable, rather than hallucinating fake measures?
                    4. Is the data source and description accurately described, when applicable?
                    Answer each of these four questions with a yes/no response, and if the answer is no, provide a justification.
                    IMPORTANT: Please format your answer into a parseable JSON object, of the format:
                    [{'question':'1', 'answer':'yes', 'justification':'...'}, {'question':'2', 'answer':'no', 'justification':'...'}, ...]
                    Ensure that this JSON object is enclosed in json``` ``` tags.
    """
    # with open(metadata_filepath, 'r') as file:
    #     file_content = file.read()
    for prompt, response in read_prompt_response_pairs(input_filepath):
        print(prompt)
        try:
            completion = llm.chat.completions.create(
                model="DeepSeek-R1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text",
                         "text": f"PROMPT: {prompt}\nRESPONSE: {response}\n"},
                        {"type": "text",
                         "text": f"[file name]: metadata.json\n[file content begin]{metadata_json}[file content end]"}
                    ]},
                ],
                stream=True
            )
            all_content = ''
            for chunk in completion:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        all_content += content
            match = re.search(r"</think>(.*)", all_content, re.DOTALL)
            if match:
                evaluation = match.group(1).strip()
                json_match = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", evaluation)
                if json_match:
                    json_string = json_match.group(1)
                    try:
                        data = json.loads(json_string)
                        final_dict = {}
                        final_dict['prompt'] = prompt
                        final_dict['response'] = response
                        final_dict['evaluation'] = data
                        results.append(final_dict)
                    except json.JSONDecodeError as e:
                        results.append({'prompt':prompt, 'response':response, 'error':str(e)})
                else:
                    results.append({'prompt': prompt, 'response': response, 'error': "No JSON object found enclosed in ```json ... ```"})
            else:
                results.append({'prompt':prompt, 'response':response, 'error':'No evaluation found'})
        except Exception as e:
            results.append({'prompt':prompt, 'response':response, 'error':str(e)})
    # with open(output_filepath, 'w', encoding='utf-8') as f:
    #     for prompt, response in results.items():
    #         formatted_entry = format_prompt_evaluation(prompt, response)
    #         f.write(formatted_entry)
    with open(output_filepath, 'w') as f:
        json.dump(results, f, indent=4)

def tally_results(input_file, output_file=None):
    with open(input_file, 'r') as file:
        data = json.load(file)
    data_list = []
    for prompt_response in data:
        prompt = prompt_response['prompt']
        evaluation = prompt_response['evaluation']
        q1 = 1 if (evaluation[0]['answer'].lower() == 'yes') else 0
        q2 = 1 if (evaluation[1]['answer'].lower() == 'yes') else 0
        q3 = 1 if (evaluation[2]['answer'].lower() == 'yes') else 0
        q4 = 1 if (evaluation[3]['answer'].lower() == 'yes') else 0
        data_list.append((prompt, q1, q2, q3, q4))
    df = pd.DataFrame(data_list, columns=['prompt', 'q1', 'q2', 'q3', 'q4'])
    if output_file is not None:
        df.to_csv(output_file, index=False)
    for col in ['q1', 'q2', 'q3', 'q4']:
        score = df[col].sum() / len(df[col])
        print(f"Score for {col}: {score}")


topics_file = 'topics.txt'
prompts_file = 'prompts_noisy.json'
prompts_filtered_file = 'prompts_noisy_filtered.json'
response_file = 'prompt_output_noisy.txt'
metadata_file = 'current_metadata_official_urls_new.csv'
evaluation_file = 'output_evaluation_noisy_with_json_metadata.json'
metrics_file = 'output_evaluation_scores_noisy_with_json_metadata.csv'

metadata_df = pd.read_csv(metadata_file)
metadata_df = metadata_df[~metadata_df['newMeasureID'].isna()]
metadata_df.set_index('newMeasureID', inplace=True)
metadata_json = metadata_df.to_json(orient='index', indent=2)

print('generating')
#generate_prompts(topics_file, prompts_file)
print('filtering')
#filter_prompts(prompts_file, prompts_filtered_file)
print('testing')
test_prompts(prompts_filtered_file, response_file, metadata_json)
print('evaluating')
evaluate_prompts(response_file, evaluation_file, metadata_json)
tally_results(evaluation_file)
