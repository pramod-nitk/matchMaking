# src/extraction.py

import json
import re
import logging
import ast
import requests
logger = logging.getLogger(__name__)

def get_server_health(base_url: str):
    # returns server health before running the API
    try:
        response = requests.get(f'{base_url}/health', timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error("Error checking server health.")
            return {"status": "error"}
    except requests.exceptions.RequestException as e:
        logger.exception(e)
        return {"status": "error"}

def post_completion(base_url: str, context: str, user_input: str):
    """
    Sends a request to the local LLM server for text completion.
    """
    prompt = f"{context}\nUser: {user_input}\nAssistant:"
    data = {
        'prompt': prompt,
        'temperature': 0.8,
        'top_k': 200,
        'top_p': 0.95,
        'n_predict': 2000,
        'stop': ["</s>", "Assistant:", "User:"]
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(f'{base_url}/completion', json=data, headers=headers)
        if response.status_code == 200:
            return response.json().get('content', "").strip()
        else:
            logger.error("Error from LLM server.")
            return "Error processing your request."
    except requests.exceptions.RequestException as e:
        logger.exception(e)
        return "Error processing your request."

def extract_json_sections(input_str: str):
    """
    Extracts JSON sections (```json ... ```) from a large string.
    """
    pattern = r'```json\n({.*?})\n```'
    matches = re.findall(pattern, input_str, re.DOTALL)
    return matches

def parse_json_sections(json_strings):
    """
    Takes a list of JSON strings, parses each, and returns a list of dictionaries.
    """
    return [json.loads(json_str) for json_str in json_strings]

def parse_resume_file(content: str):
    """
    Parse the resume file that may contain either:
      1) A single Python dict block (like the first example).
      2) Multiple sections (I. Personal Information:, II. Education:, etc.) each followed by a dict.

    Returns a Python dictionary object.
    """
    # 1) Try parsing as a single dictionary (the "one-block" style).
    try:
        # ast.literal_eval can handle single quotes and Python dict syntax
        single_dict = ast.literal_eval(content)
        if isinstance(single_dict, dict):
            # Successfully parsed as one dictionary, return it
            return single_dict
    except (SyntaxError, ValueError):
        # If it fails, we proceed to the multi-section approach
        pass


    pattern = r'^([IVX]+)\.\s+(.*?):\s*\n(.*?)(?=^[IVX]+\.\s+|$)'
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL)

    sections = regex.findall(content)
    if not sections:
        # If we still can't find any sections, it might mean the file isn't following
        # the expected format at all. You may decide to raise an error or return None.
        raise ValueError("Could not parse file as single dict or multi-section format.")

    # Construct a final dictionary in which:
    #  keys = heading text (e.g. "Personal Information", "Education", "Projects")
    #  values = dict parsed from that section
    final_data = {}
    for roman_numeral, heading_text, dict_block in sections:
        dict_block = dict_block.strip()

        # Attempt to parse this block as a Python dictionary
        try:
            section_dict = ast.literal_eval(dict_block)
        except (SyntaxError, ValueError) as e:
            # If it fails to parse, store an error or skip. Adjust as you prefer.
            print(f"Warning: Could not parse section '{heading_text}' due to: {e}")
            section_dict = None

        # You might want the key to be exactly heading_text, e.g. "Personal Information"
        final_data[heading_text] = section_dict

    return final_data

def parse_resume_file_2(content):
    ## Parse json from string
    try:
        pattern = r'```json\n({.*?})\n```'
        matches = re.findall(pattern, content, re.DOTALL)
        return json.loads(matches[0].replace("'",'"'))
    except Exception:
        print(f"Warning: Could not parse section")
        return None
