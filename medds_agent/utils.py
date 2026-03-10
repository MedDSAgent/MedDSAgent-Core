import re
from typing import Union, Dict

def apply_prompt_template(prompt_template:str, text_content:Union[str, Dict[str,str]]) -> str:
    """
    This method applies text_content to prompt_template and returns a prompt.

    Parameters:
    ----------
    prompt_template : str
        the prompt template with placeholders {{<placeholder name>}}.
    text_content : Union[str, Dict[str,str]]
        the input text content to put in prompt template. 
        If str, the prompt template must has only 1 placeholder {{<placeholder name>}}, regardless of placeholder name.
        If dict, all the keys must be included in the prompt template placeholder {{<placeholder name>}}. All values must be str.

    Returns : str
        a user prompt.
    """
    if not isinstance(text_content, (str, dict)):
        raise ValueError(f"text_content must be str or dict[str, str]. Received type: {type(text_content)}")

    pattern = re.compile(r'{{(.*?)}}')
    if isinstance(text_content, str):
        matches = pattern.findall(prompt_template)
        if len(matches) != 1:
            raise ValueError("When text_content is str, the prompt template must has exactly 1 placeholder {{<placeholder name>}}.")
        text = re.sub(r'\\', r'\\\\', text_content)
        prompt = pattern.sub(text, prompt_template)

    elif isinstance(text_content, dict):
        # Check if all values are str
        if not all([isinstance(v, str) for v in text_content.values()]):
            raise ValueError("All values in text_content must be str.")
        # Check if all keys are in the prompt template
        placeholders = pattern.findall(prompt_template)
        if len(placeholders) != len(text_content):
            raise ValueError(f"Expect text_content ({len(text_content)}) and prompt template placeholder ({len(placeholders)}) to have equal size.")
        if not all([k in placeholders for k, _ in text_content.items()]):
            raise ValueError(f"All keys in text_content ({text_content.keys()}) must match placeholders in prompt template ({placeholders}).")

        prompt = pattern.sub(lambda match: re.sub(r'\\', r'\\\\', text_content[match.group(1)]), prompt_template)

    return prompt