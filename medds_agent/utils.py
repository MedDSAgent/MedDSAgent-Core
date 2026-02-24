import io
import re
from typing import Union, Dict, List, Any
import json
import base64
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import dill

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


def serialize_object(obj: Any) -> Any:
    """ 
    Recursive function to any object a JSON-serializable object. 
    
    Handles:
    - Basic types (str, int, float, bool, None)
    - Lists and tuples
    - Dictionaries
    - PIL Images (converted to base64 strings)
    - Matplotlib Figures (converted to base64 strings)
    - Pandas DataFrames (converted to markdown tables)
    - Custom objects (converted to their __dict__)
    - Other types converted to strings
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        # Try to parse string as JSON if it looks like a JSON object/array
        if isinstance(obj, str):
            try:
                if (obj.startswith("{") and obj.endswith("}")) or (obj.startswith("[") and obj.endswith("]")):
                    parsed = json.loads(obj)
                    return serialize_object(parsed)
            except json.JSONDecodeError:
                pass
        return obj
    elif isinstance(obj, (list, tuple)):
        return [serialize_object(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): serialize_object(v) for k, v in obj.items()}
    # PIL Image -> Base64
    elif Image and isinstance(obj, Image.Image): 
        try:
            buffered = io.BytesIO()
            obj.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{img_str}"
        except Exception:
            return "[Error serializing Image]"
    # Matplotlib Figure -> Base64
    elif plt and hasattr(obj, 'savefig'):
        try:
            buffered = io.BytesIO()
            obj.savefig(buffered, format='png')
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{img_str}"
        except Exception:
            return "[Error serializing Plot]"

    # Pandas DataFrame -> Markdown Table
    elif pd and isinstance(obj, pd.DataFrame):
        try:
            return obj.to_markdown(index=True)
        except Exception:
            return str(obj)
    elif hasattr(obj, "__dict__"):
        # For custom object, convert their __dict__ to a serializable format
        return {"_type": obj.__class__.__name__, **{k: serialize_object(v) for k, v in obj.__dict__.items()}}
    else:
        # For any other type, convert to string
        return str(obj)
    
def serialize_python_state(state: Dict[str, Any]) -> bytes:
    """
    Serializes the Python state dictionary using dill.
    Returns bytes suitable for BLOB storage.
    """
    if not state:
        return b""
    try:
        return dill.dumps(state)
    except Exception as e:
        print(f"Warning: Could not serialize python state: {e}")
        return b""

def deserialize_python_state(data: bytes) -> Any:
    """
    Deserializes the Python state from bytes.
    """
    if not data:
        return None
    try:
        return dill.loads(data)
    except Exception as e:
        print(f"Warning: Could not restore python state: {e}")
        return {}