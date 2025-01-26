import json
def parse_resume_file_chatapi(content):
    try:
        content = content.replace('<|end_of_turn|>', '').replace("'",'"')
        parsed = json.loads(str(content))
        return parsed
    except Exception as e:
        print(f"Warning: Could not parse section", e)
        return None