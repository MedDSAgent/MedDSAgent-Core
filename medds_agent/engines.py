from llm_inference_engine import (
    # Configs
    LLMConfig,
    BasicLLMConfig,
    ReasoningLLMConfig,
    Qwen3LLMConfig,
    OpenAIReasoningLLMConfig,
    
    # Base Engine
    InferenceEngine,
    
    # Concrete Engines
    OllamaInferenceEngine,
    HuggingFaceHubInferenceEngine,
    LiteLLMInferenceEngine,
    OpenAIInferenceEngine,
    AzureOpenAIInferenceEngine,
    OpenAICompatibleInferenceEngine,
    VLLMInferenceEngine,
    SGLangInferenceEngine,
    OpenRouterInferenceEngine
)

from llm_inference_engine.utils import MessagesLogger

import warnings
from typing import Callable, List, Dict, Optional, Any

from typing import List, Dict, Callable, Any

def slide_window_messages_filter(messages: List[Dict[str, str]], 
                                 max_tokens: int, 
                                 tokenizer: Callable[[str], Any], 
                                 truncate_hint: str = "[CONTENT TRUNCATED FOR CONTEXT LENGTH]") -> List[Dict[str, str]]:
    """
    A message filter that retains the first message (system prompt), and then 
    applies a sliding window to the remaining messages (first N, last N).
    
    If messages are skipped, a single placeholder message is inserted 
    containing the truncate_hint.
    """
    if not messages:
        return messages
    
    # Validation: Ensure all contents are strings
    for message in messages:
        if not isinstance(message.get('content', ''), str):
            raise ValueError(f"All message contents must be strings: {message.get('content', '')}")

    included_message_indexes = set()
    # Iterate forward to include messages until we reach (max_tokens/2)
    current_tokens = 0
    for i, message in enumerate(messages):
        new_tokens = len(tokenizer(message['content']))
        if message['role'] == 'system':
            included_message_indexes.add(i)  # Always include the first message (system prompt)
            current_tokens += new_tokens
            continue
        
        # Calculate total tokens if we include this message
        if current_tokens + new_tokens <= max_tokens / 2:
            included_message_indexes.add(i)
            current_tokens += new_tokens
        else:
            break

    # Iterate backward to include messages until we reach (max_tokens/2)
    current_tokens = 0
    for i in range(len(messages) - 1, -1, -1):        
        new_tokens = len(tokenizer(messages[i]['content']))
        if current_tokens + new_tokens <= max_tokens / 2:
            included_message_indexes.add(i)
            current_tokens += new_tokens
        else:
            break

    # Add messages to output
    output_messages = []
    for i, message in enumerate(messages):
        if i in included_message_indexes:
            output_messages.append(message)
        else:
            output_messages.append({'role': message['role'], 'content': truncate_hint})

    return output_messages


class AgentBasicLLMConfig(BasicLLMConfig):
    def __init__(self, max_new_tokens:int=16384, temperature:float=0.7, tokenizer: Callable[[str], Any]=None, 
                 max_input_tokens:int=120000, **kwargs):
        super().__init__(max_new_tokens=max_new_tokens, temperature=temperature, **kwargs)
        # If no tokenizer is provided, use a simple whitespace tokenizer
        self.tokenizer = tokenizer if tokenizer is not None else lambda x: x.split()
        self.max_input_tokens = max_input_tokens

    def preprocess_messages(self, messages:List[Dict[str,str]]) -> List[Dict[str,str]]:
        return slide_window_messages_filter(messages, self.max_input_tokens, self.tokenizer)
    
class AgentReasoningLLMConfig(ReasoningLLMConfig):
    def __init__(self, thinking_token_start="<think>", thinking_token_end="</think>", tokenizer: Callable[[str], Any]=None, 
                 max_input_tokens:int=120000, **kwargs):
        super().__init__(thinking_token_start=thinking_token_start, thinking_token_end=thinking_token_end, **kwargs)
        # If no tokenizer is provided, use a simple whitespace tokenizer
        self.tokenizer = tokenizer if tokenizer is not None else lambda x: x.split()
        self.max_input_tokens = max_input_tokens

    def preprocess_messages(self, messages:List[Dict[str,str]]) -> List[Dict[str,str]]:
        return slide_window_messages_filter(messages, self.max_input_tokens, self.tokenizer)
    
class AgentQwen3LLMConfig(Qwen3LLMConfig):
    def __init__(self, thinking_mode:bool=True, tokenizer: Callable[[str], Any]=None, max_input_tokens:int=120000, **kwargs):
        """
        The Qwen3 **hybrid thinking** LLM configuration. 
        For Qwen3 thinking 2507, use ReasoningLLMConfig instead; for Qwen3 Instruct, use BasicLLMConfig instead.

        Parameters:
        ----------
        thinking_mode : bool, Optional
            if True, a special token "/think" will be placed after each system and user prompt. Otherwise, "/no_think" will be placed.
        tokenizer: Callable[[str], Any], Optional
            a tokenizer function that takes a string and returns a list of tokens.
        max_input_tokens: int, Optional
            the maximum number of input tokens.
        """
        super().__init__(**kwargs)
        self.thinking_mode = thinking_mode
        # If no tokenizer is provided, use a simple whitespace tokenizer
        self.tokenizer = tokenizer if tokenizer is not None else lambda x: x.split()
        self.max_input_tokens = max_input_tokens

    def preprocess_messages(self, messages:List[Dict[str,str]]) -> List[Dict[str,str]]:
        """
        Append a special token to the system and user prompts.
        The token is "/think" if thinking_mode is True, otherwise "/no_think".

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        
        Returns:
        -------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        """
        thinking_token = "/think" if self.thinking_mode else "/no_think"
        new_messages = []
        for message in messages:
            if message['role'] in ['system', 'user']:
                new_message = {'role': message['role'], 'content': f"{message['content']} {thinking_token}"}
            else:
                new_message = {'role': message['role'], 'content': message['content']}

            new_messages.append(new_message)

        return slide_window_messages_filter(new_messages, self.max_input_tokens, self.tokenizer)
    
class AgentOpenAIReasoningLLMConfig(OpenAICompatibleInferenceEngine):
    def __init__(self, reasoning_effort:str=None, tokenizer: Callable[[str], Any]=None, max_input_tokens:int=120000, **kwargs):
        """
        The OpenAI "o" series configuration.
        1. The reasoning effort as one of {"low", "medium", "high"}.
            For models that do not support setting reasoning effort (e.g., o1-mini, o1-preview), set to None.
        2. The temperature parameter is not supported and will be ignored.
        3. The system prompt is not supported and will be concatenated to the next user prompt.

        Parameters:
        ----------
        reasoning_effort : str, Optional
            the reasoning effort. Must be one of {"low", "medium", "high"}. Default is "low".
        """
        super().__init__(**kwargs)
        # If no tokenizer is provided, use a simple whitespace tokenizer
        self.tokenizer = tokenizer if tokenizer is not None else lambda x: x.split()
        self.max_input_tokens = max_input_tokens
        
        if reasoning_effort is not None:
            if reasoning_effort not in ["low", "medium", "high"]:
                raise ValueError("reasoning_effort must be one of {'low', 'medium', 'high'}.")

            self.reasoning_effort = reasoning_effort
            self.params["reasoning_effort"] = self.reasoning_effort

        if "temperature" in self.params:
            warnings.warn("Reasoning models do not support temperature parameter. Will be ignored.", UserWarning)
            self.params.pop("temperature")

    def preprocess_messages(self, messages:List[Dict[str,str]]) -> List[Dict[str,str]]:
        """
        Concatenate system prompts to the next user prompt.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        
        Returns:
        -------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        """
        system_prompt_holder = ""
        new_messages = []
        for i, message in enumerate(messages):
            # if system prompt, store it in system_prompt_holder
            if message['role'] == 'system':
                system_prompt_holder = message['content']
            # if user prompt, concatenate it with system_prompt_holder
            elif message['role'] == 'user':
                if system_prompt_holder:
                    new_message = {'role': message['role'], 'content': f"{system_prompt_holder} {message['content']}"}
                    system_prompt_holder = ""
                else:
                    new_message = {'role': message['role'], 'content': message['content']}

                new_messages.append(new_message)
            # if assistant/other prompt, do nothing
            else:
                new_message = {'role': message['role'], 'content': message['content']}
                new_messages.append(new_message)

        return slide_window_messages_filter(new_messages, self.max_input_tokens, self.tokenizer)