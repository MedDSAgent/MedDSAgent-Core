import json
import asyncio
import logging
import traceback
from typing import List, Dict, Any, Generator, AsyncGenerator, Union
from datetime import datetime
import importlib.resources

logger = logging.getLogger(__name__)
from rich.console import Group
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from llm_inference_engine import InferenceEngine
from medds_agent.history import SystemStep, UserStep, AgentStep, ObservationStep, History
from medds_agent.memory import AgentMemory, FullHistoryAgentMemory
from medds_agent.tools import Tool, FinalResponseTool
from medds_agent.utils import apply_prompt_template
import uuid


class Agent:
    def __init__(self, llm_engine: InferenceEngine, history:History, agent_memory: AgentMemory=None,
                 specialty_prompt:str=None, tools:List[Tool]=None, verbose:bool=False, max_retries:int=8):
        """
        Agent class that executes the Action-Observation loop.

        Parameters:
        -----------
        llm_engine: InferenceEngine
            The LLM inference engine to use for generating responses.
        history: History
            The history object to store action history. History is shared by all agents.
        agent_memory: AgentMemory, Optional
            The memory management strategy for the agent. Default is FullHistoryAgentMemory.
        specialty_prompt: str, Optional
            Domain or task-specific instructions that can be appended to the system prompt for more context.
        tools: List[Tool], optional
            List of custom tools to be used by the agent.
        max_retries: int, optional
            Maximum number of retries for LLM response generation after incorrect outputs.
            Retries within the max will NOT be stored in history. Retries exhausted will be returned as final response.
        verbose: bool, optional
            Whether to print verbose logs during agent execution.
        """
        # Agent ID
        self.agent_id = str(uuid.uuid4())

        # Define LLM engine
        self.llm_engine = llm_engine
        self.max_retries = max_retries
        self.verbose = verbose

        # Define agent memory
        if agent_memory is None:
            self.agent_memory = FullHistoryAgentMemory()
        else:
            self.agent_memory = agent_memory

        # Initialize Rich Console
        self.console = Console()

        # Load system prompt
        file_path_obj = importlib.resources.files('medds_agent.asset.prompt_templates').joinpath("System_prompt_template.md")
        with importlib.resources.as_file(file_path_obj) as file_path:
            with open(file_path, "r") as file:
                self.prompt_template = file.read()
        self.system_prompt = apply_prompt_template(self.prompt_template, {})
        self.agent_memory.set_system_prompt(self.system_prompt, specialty_prompt)

        # Hook to history
        self.history = history

        # Define tools — always append FinalResponseTool as the end-of-round signal
        self.tools = (tools if tools is not None else []) + [FinalResponseTool()]

    def reset(self):
        """
        Reset the agent's memory while keeping system step.
        """
        self.agent_memory.reset()

    def _log_step(self, step):
        """
        Internal method to print steps with Rich formatting if verbose is True.
        """
        if not self.verbose:
            return

        if isinstance(step, UserStep):
            self.console.print(Panel(step.user_input, title="User Input", style="green"))

        elif isinstance(step, AgentStep):
            # Interim narration (if any)
            if step.response:
                self.console.print(Panel(Markdown(step.response), title="Agent Response", style="magenta"))

            # Tool calls
            for t in step.tools:
                tool_name = t["tool_name"]
                tool_args = t["tool_args"]
                tool_title = t.get("tool_title", "")
                panel_title = tool_title if tool_title else f"Tool Call: {tool_name}"
                if tool_name in ("PythonExecutor", "RExecutor"):
                    args_dict = json.loads(tool_args)
                    lang = "python" if tool_name == "PythonExecutor" else "r"
                    syntax = Syntax(args_dict['code'], lang, theme="monokai", line_numbers=True)
                    content = Group(f"[bold]Tool:[/bold] {tool_name}", syntax)
                    self.console.print(Panel(content, title=panel_title, style="cyan"))
                elif tool_name == "final_response":
                    args_dict = json.loads(tool_args)
                    response_text = args_dict.get("response", "")
                    self.console.print(Panel(Markdown(response_text), title="Final Response", style="bold purple"))
                else:
                    self.console.print(Panel(f"Tool: {tool_name}\nParameters: {tool_args}", title=panel_title, style="cyan"))

        elif isinstance(step, ObservationStep):
            for to in step.tool_outputs:
                content = f"Output ({to['tool_name']}):\n{to['output']}"
                self.console.print(Panel(str(content).strip(), title="Tool Output", style="white on black", border_style="bold white"))

    def _record_error_as_final(self, error_message: str):
        """
        Record a system error as a final AgentStep.
        Used when all retries are exhausted.
        """
        agent_step = AgentStep(
            agent_id=self.agent_id,
            start_time=datetime.now(),
            end_time=datetime.now(),
            response=error_message,
            tools=[],
            is_final=True,
        )
        if self.verbose:
            self._log_step(agent_step)
        self.history.add_step(agent_step)
        self.agent_memory.on_step_added(agent_step)

    def _process_llm_response(self, llm_response: Dict[str, Any], stream: bool = False) -> Union[bool, Generator]:
        """
        Process an LLM response: record one AgentStep and (if tools ran) one ObservationStep.

        End-of-round rules:
          - Tool calls present, final_response alone → is_final=True, end round.
          - Tool calls present, final_response mixed with others → raise ValueError (retry).
          - Tool calls present, no final_response → execute tools, continue.
          - No tool calls → raise ValueError (retry; LLM must always call a tool).

        Parameters:
        -----------
        stream: bool
            If True, returns a generator that yields {'type': 'tool_output'|'response'|'final_decision'}.
            If False, consumes the generator internally and returns a boolean (is_final).
        """
        def _generator_logic():
            response_text = llm_response.get('response', '') or ""
            tool_calls = llm_response.get('tool_calls', []) or []

            if not tool_calls:
                raise ValueError(
                    "No tool calls received. You must always respond by calling a tool. "
                    "Use final_response when you are ready to deliver your answer."
                )

            # Parse all calls once
            parsed = []
            for tc in tool_calls:
                tc_name = tc['name'] or ""
                arguments = json.loads(tc['arguments'])
                tool = next((t for t in self.tools if t.name == tc_name), None)
                parsed.append({"name": tc_name, "arguments": arguments, "tool": tool})

            final_calls = [p for p in parsed if p["name"] == "final_response"]
            real_calls  = [p for p in parsed if p["name"] != "final_response"]

            if final_calls and real_calls:
                raise ValueError(
                    "final_response must be called alone. "
                    "Do not mix it with other tool calls in the same response."
                )

            is_final = bool(final_calls)

            # Build tools list for AgentStep
            tools_list = [
                {
                    "tool_name": p["name"],
                    "tool_args": json.dumps(p["arguments"], indent=2),
                    "tool_title": p["tool"].get_title(p["arguments"]) if p["tool"] else "",
                }
                for p in parsed
            ]

            # Record single AgentStep for the whole LLM turn
            agent_step = AgentStep(
                agent_id=self.agent_id,
                start_time=datetime.now(),
                end_time=datetime.now(),
                response=response_text,
                tools=tools_list,
                is_final=is_final,
            )
            if self.verbose:
                self._log_step(agent_step)
            self.history.add_step(agent_step)
            self.agent_memory.on_step_added(agent_step)

            if is_final:
                # Deliver final_response response to the frontend as a response event
                final_response = parsed[0]["arguments"].get("response", "")
                yield {"type": "response", "data": final_response}
                yield {"type": "final_decision", "is_final": True}
                return

            # Execute real tools, collect outputs
            tool_outputs = []
            for p in real_calls:
                tc_name, arguments, tool = p["name"], p["arguments"], p["tool"]
                if tool is None:
                    tool_output = f"Error: Tool '{tc_name}' not found."
                else:
                    try:
                        tool_output = str(tool.execute(arguments))
                    except Exception as e:
                        tool_output = f"Error executing tool '{tc_name}': {str(e)}"
                tool_outputs.append({"tool_name": tc_name, "output": tool_output})
                yield {"type": "tool_output", "data": tool_output}

            # Record single ObservationStep for all outputs
            observation_step = ObservationStep(
                agent_id=self.agent_id,
                start_time=datetime.now(),
                end_time=datetime.now(),
                tool_outputs=tool_outputs,
            )
            if self.verbose:
                self._log_step(observation_step)
            self.history.add_step(observation_step)
            self.agent_memory.on_step_added(observation_step)

            yield {"type": "final_decision", "is_final": False}

        # --- Main Dispatch Logic ---
        gen = _generator_logic()

        if stream:
            return gen
        else:
            end_round = False
            for event in gen:
                if event['type'] == 'final_decision':
                    end_round = event['is_final']
            return end_round

    async def _process_llm_response_async(self, llm_response: Dict[str, Any], stream: bool = False) -> Union[bool, AsyncGenerator]:
        """
        Async version of _process_llm_response. Tool execution runs in thread pool to avoid blocking.

        Parameters:
        -----------
        stream: bool
            If True, returns an async generator that yields {'type': 'tool_output'|'response'|'final_decision'}.
            If False, processes internally and returns a boolean (is_final).
        """
        async def _async_generator_logic():
            response_text = llm_response.get('response', '') or ""
            tool_calls = llm_response.get('tool_calls', []) or []

            if not tool_calls:
                raise ValueError(
                    "No tool calls received. You must always respond by calling a tool. "
                    "Use final_response when you are ready to deliver your answer and end the round."
                )

            # Parse all calls once
            parsed = []
            for tc in tool_calls:
                tc_name = tc['name'] or ""
                arguments = json.loads(tc['arguments'])
                tool = next((t for t in self.tools if t.name == tc_name), None)
                parsed.append({"name": tc_name, "arguments": arguments, "tool": tool})

            final_calls = [p for p in parsed if p["name"] == "final_response"]
            real_calls  = [p for p in parsed if p["name"] != "final_response"]

            if final_calls and real_calls:
                raise ValueError(
                    "final_response must be called alone. "
                    "Do not mix it with other tool calls in the same response."
                )

            is_final = bool(final_calls)

            # Build tools list for AgentStep
            tools_list = [
                {
                    "tool_name": p["name"],
                    "tool_args": json.dumps(p["arguments"], indent=2),
                    "tool_title": p["tool"].get_title(p["arguments"]) if p["tool"] else "",
                }
                for p in parsed
            ]

            # Record single AgentStep for the whole LLM turn
            agent_step = AgentStep(
                agent_id=self.agent_id,
                start_time=datetime.now(),
                end_time=datetime.now(),
                response=response_text,
                tools=tools_list,
                is_final=is_final,
            )
            if self.verbose:
                self._log_step(agent_step)
            self.history.add_step(agent_step)
            await self.agent_memory.on_step_added_async(agent_step)

            if is_final:
                # Deliver final_response response to the frontend as a response event
                final_response = parsed[0]["arguments"].get("response", "")
                yield {"type": "response", "data": final_response}
                yield {"type": "final_decision", "is_final": True}
                return

            # Execute real tools asynchronously, collect outputs
            tool_outputs = []
            for p in real_calls:
                tc_name, arguments, tool = p["name"], p["arguments"], p["tool"]
                if tool is None:
                    tool_output = f"Error: Tool '{tc_name}' not found."
                else:
                    try:
                        tool_output = await asyncio.to_thread(tool.execute, arguments)
                        tool_output = str(tool_output)
                    except Exception as e:
                        tool_output = f"Error executing tool '{tc_name}': {str(e)}"
                tool_outputs.append({"tool_name": tc_name, "output": tool_output})
                yield {"type": "tool_output", "data": tool_output}

            # Record single ObservationStep for all outputs
            observation_step = ObservationStep(
                agent_id=self.agent_id,
                start_time=datetime.now(),
                end_time=datetime.now(),
                tool_outputs=tool_outputs,
            )
            if self.verbose:
                self._log_step(observation_step)
            self.history.add_step(observation_step)
            await self.agent_memory.on_step_added_async(observation_step)

            yield {"type": "final_decision", "is_final": False}

        # --- Main Dispatch Logic ---
        if stream:
            return _async_generator_logic()
        else:
            final_result = False
            async for event in _async_generator_logic():
                if event['type'] == 'final_decision':
                    final_result = event['is_final']
            return final_result

    def run_stream(self, user_input: str) -> Generator[Dict[str, str], None, None]:
        """
        Run the agent with streaming output (sync).

        Parameters:
        -----------
        user_input: str
            The user input to process.

        Yields:
        -------
        Dict[str, str]
            Streamed output from the LLM. Dictionary with 'type' and 'data' keys.
            types can be 'response', 'tool_calls', 'tool_output'.
        """
        # Add user step to history
        user_step = UserStep(start_time=datetime.now(),
                             end_time=datetime.now(),
                             user_input=user_input)
        self.history.add_step(user_step)
        self.agent_memory.on_step_added(user_step)

        # Log user step
        if self.verbose:
            self._log_step(user_step)

        while True:
            messages = self.agent_memory.get_messages(self.history)
            tools_schema = [tool.get_tool_call_schema() for tool in self.tools]

            for attempt in range(self.max_retries):
                accumulated_response = ""
                accumulated_tool_calls = []

                try:
                    stream_generator = self.llm_engine.chat_stream(
                        messages=messages,
                        tools=tools_schema
                    )

                    # 1. Stream the LLM text generation
                    for chunk in stream_generator:
                        if isinstance(chunk, dict):
                            chunk_type = chunk.get('type', '')
                            chunk_data = chunk.get('data', '')

                            if chunk_type == 'response':
                                accumulated_response += chunk_data
                                yield {"type": "response", "data": chunk_data}
                            elif chunk_type == 'tool_calls':
                                accumulated_tool_calls = chunk_data
                                yield {"type": "tool_calls", "data": chunk_data}

                    # 2. Process Execution (using stream=True)
                    llm_response = {
                        "response": accumulated_response,
                        "tool_calls": accumulated_tool_calls
                    }

                    is_final_answer = False

                    # Consume the tool execution generator
                    execution_gen = self._process_llm_response(llm_response, stream=True)

                    for event in execution_gen:
                        if event['type'] == 'tool_output':
                            # Relay output to the frontend
                            yield event
                        elif event['type'] == 'final_decision':
                            is_final_answer = event['is_final']

                    # Success, break retry loop
                    break

                except Exception as e:
                    # Capture the validation error or execution error
                    if attempt < self.max_retries - 1:
                        # Append the error to the message history so the model sees it in the next loop
                        messages.append({"role": "assistant", "content": accumulated_response})
                        messages.append({"role": "user", "content": f"Error: {str(e)}. Please regenerate the response fixing this error."})
                    else:
                        error_msg = f"Error: {str(e)}. Unable to generate a valid response."
                        yield {"type": "response", "data": f"\n\n[System Error: {error_msg}]"}
                        self._record_error_as_final(error_msg)
                        return

            if is_final_answer:
                return

    def run(self, user_input: str) -> None:
        """
        Run the agent with the given user input (sync, non-streaming).

        Parameters:
        -----------
        user_input: str
            The user input to process.
        """
        # Add user step to history
        user_step = UserStep(start_time=datetime.now(),
                             end_time=datetime.now(),
                             user_input=user_input)
        self.history.add_step(user_step)
        self.agent_memory.on_step_added(user_step)

        # Log user step
        if self.verbose:
            self._log_step(user_step)

        # Run agent loop
        while True:
            messages = self.agent_memory.get_messages(self.history)
            for attempt in range(self.max_retries):
                # Standard blocking chat call
                llm_response = self.llm_engine.chat(
                    messages=messages,
                    verbose=False,
                    tools=[tool.get_tool_call_schema() for tool in self.tools]
                )

                try:
                    is_final_answer = self._process_llm_response(llm_response, stream=False)

                    # Success, break retry loop
                    break

                except Exception as e:
                    ## Debugging: print the exception
                    print(f"Exception during LLM response processing: {str(e)}")

                    if attempt < self.max_retries - 1:
                        response_text = llm_response.get('response', '')
                        response_text = response_text if response_text else ""
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content": f"Error: {str(e)}. Please regenerate the response fixing this error."})
                    else:
                        error_msg = f"Error: {str(e)}. Unable to generate a valid response."
                        self._record_error_as_final(error_msg)
                        return

            if is_final_answer:
                return

    async def run_async(self, user_input: str) -> None:
        """
        Run the agent with the given user input (async, non-streaming).
        Uses chat_async() for non-blocking LLM calls.

        Parameters:
        -----------
        user_input: str
            The user input to process.
        """
        # Add user step to history
        user_step = UserStep(start_time=datetime.now(),
                             end_time=datetime.now(),
                             user_input=user_input)
        self.history.add_step(user_step)
        await self.agent_memory.on_step_added_async(user_step)

        # Log user step
        if self.verbose:
            self._log_step(user_step)

        # Run agent loop
        while True:
            messages = self.agent_memory.get_messages(self.history)
            tools_schema = [tool.get_tool_call_schema() for tool in self.tools]

            for attempt in range(self.max_retries):
                try:
                    # Async non-blocking chat call
                    llm_response = await self.llm_engine.chat_async(
                        messages=messages,
                        tools=tools_schema
                    )

                    # Process response (async version)
                    is_final_answer = await self._process_llm_response_async(llm_response, stream=False)

                    # Success, break retry loop
                    break

                except Exception as e:
                    if attempt < self.max_retries - 1:
                        response_text = llm_response.get('response', '') if 'llm_response' in dir() else ''
                        response_text = response_text if response_text else ""
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content": f"Error: {str(e)}. Please regenerate the response fixing this error."})
                    else:
                        error_msg = f"Error: {str(e)}. Unable to generate a valid response."
                        self._record_error_as_final(error_msg)
                        return

            if is_final_answer:
                return

    async def run_async_stream(self, user_input: str) -> AsyncGenerator[Dict[str, str], None]:
        """
        Run the agent with streaming output (async).
        Uses chat_async_stream() for non-blocking streaming LLM calls.

        Parameters:
        -----------
        user_input: str
            The user input to process.

        Yields:
        -------
        Dict[str, str]
            Streamed output from the LLM. Dictionary with 'type' and 'data' keys.
            types can be 'response', 'tool_calls', 'tool_output'.
        """
        # Add user step to history
        user_step = UserStep(start_time=datetime.now(),
                             end_time=datetime.now(),
                             user_input=user_input)
        self.history.add_step(user_step)
        await self.agent_memory.on_step_added_async(user_step)

        # Log user step
        if self.verbose:
            self._log_step(user_step)

        while True:
            messages = self.agent_memory.get_messages(self.history)
            tools_schema = [tool.get_tool_call_schema() for tool in self.tools]

            for attempt in range(self.max_retries):
                accumulated_response = ""
                accumulated_tool_calls = []

                try:
                    # Async streaming chat call
                    stream_generator = await self.llm_engine.chat_async_stream(
                        messages=messages,
                        tools=tools_schema
                    )

                    # 1. Stream the LLM text generation
                    async for chunk in stream_generator:
                        if isinstance(chunk, dict):
                            chunk_type = chunk.get('type', '')
                            chunk_data = chunk.get('data', '')

                            if chunk_type == 'response':
                                accumulated_response += chunk_data
                                yield {"type": "response", "data": chunk_data}
                            elif chunk_type == 'tool_calls':
                                accumulated_tool_calls = chunk_data
                                yield {"type": "tool_calls", "data": chunk_data}

                    # 2. Process Execution (using async stream=True)
                    llm_response = {
                        "response": accumulated_response,
                        "tool_calls": accumulated_tool_calls
                    }

                    is_final_answer = False

                    # Consume the async tool execution generator
                    execution_gen = await self._process_llm_response_async(llm_response, stream=True)

                    async for event in execution_gen:
                        if event['type'] == 'tool_output':
                            # Relay output to the frontend
                            yield event
                        elif event['type'] == 'final_decision':
                            is_final_answer = event['is_final']

                    # Success, break retry loop
                    break

                except Exception as e:
                    ## Debugging: print the exception
                    print(f"Exception during LLM response processing: {str(e)}")

                    # Capture the validation error or execution error
                    if attempt < self.max_retries - 1:
                        # Append the error to the message history so the model sees it in the next loop
                        messages.append({"role": "assistant", "content": accumulated_response})
                        messages.append({"role": "user", "content": f"Error: {str(e)}. Please regenerate the response fixing this error."})
                    else:
                        error_msg = f"Error: {str(e)}. Unable to generate a valid response."
                        yield {"type": "response", "data": f"\n\n[System Error: {error_msg}]"}
                        self._record_error_as_final(error_msg)
                        return

            if is_final_answer:
                return
            
    async def run_event_stream_async(self, user_input: str) -> AsyncGenerator[Dict[str, str], None]:
        """
        Run the agent with "Block Streaming" (Event-based).
        
        Instead of streaming tokens, this yields full blocks of data as they become available:
        1. Full Text Response (Plan)
        2. Tool Call Details
        3. Tool Outputs (Result)

        Parameters:
        -----------
        user_input: str
            The user input to process.

        Yields:
        -------
        Dict[str, str]
            Event dictionary with 'type' and 'data'.
            Types: 'response' (full text), 'tool_calls' (list), 'tool_output' (str).
        """
        # Add user step to history
        user_step = UserStep(start_time=datetime.now(),
                             end_time=datetime.now(),
                             user_input=user_input)
        self.history.add_step(user_step)
        await self.agent_memory.on_step_added_async(user_step)

        if self.verbose:
            self._log_step(user_step)

        while True:
            messages = self.agent_memory.get_messages(self.history)
            tools_schema = [tool.get_tool_call_schema() for tool in self.tools]

            for attempt in range(self.max_retries):
                try:
                    # 1. Non-blocking Chat Call (Wait for full response)
                    llm_response = await self.llm_engine.chat_async(
                        messages=messages,
                        tools=tools_schema
                    )
                    
                    # 2. Yield interim text only when tool calls are also present.
                    #    Text-only responses are retry cases — don't stream them.
                    response_text = llm_response.get('response', '') or ''
                    tool_calls = llm_response.get('tool_calls', []) or []
                    if response_text and tool_calls:
                        logger.debug(
                            "[run_event_stream_async] attempt=%d: yielding interim response_text "
                            "(len=%d) alongside %d tool call(s).",
                            attempt, len(response_text), len(tool_calls)
                        )
                        yield {"type": "response", "data": response_text}

                    # 3. Yield Tool Calls preview (exclude final_response — internal mechanism)
                    visible_tool_calls = [tc for tc in tool_calls if tc.get('name') != 'final_response']
                    if visible_tool_calls:
                        enriched = []
                        for tc in visible_tool_calls:
                            try:
                                args = json.loads(tc.get('arguments', '{}'))
                            except Exception:
                                args = {}
                            tool = next((t for t in self.tools if t.name == tc.get('name')), None)
                            enriched.append({**tc, 'tool_title': tool.get_title(args) if tool else ''})
                        yield {"type": "tool_calls", "data": enriched}

                    # 4. Process Execution (executes tools, updates history, yields outputs)
                    is_final_answer = False
                    execution_gen = await self._process_llm_response_async(llm_response, stream=True)

                    async for event in execution_gen:
                        if event['type'] in ('tool_output', 'response'):
                            yield event
                        elif event['type'] == 'final_decision':
                            is_final_answer = event['is_final']

                    logger.debug(
                        "[run_event_stream_async] attempt=%d: execution complete. is_final_answer=%s",
                        attempt, is_final_answer
                    )
                    # Success, break retry loop
                    break

                except Exception as e:
                    logger.warning(
                        "[run_event_stream_async] attempt=%d: exception caught.\nException: %s\n%s",
                        attempt, str(e), traceback.format_exc()
                    )
                    # Handle Errors
                    if attempt < self.max_retries - 1:
                        accumulated_response = llm_response.get('response', '') if 'llm_response' in dir() else ''
                        messages.append({"role": "assistant", "content": accumulated_response})
                        messages.append({"role": "user", "content": f"Error: {str(e)}. Please regenerate the response fixing this error."})
                    else:
                        error_msg = f"Error: {str(e)}. Unable to generate a valid response."
                        yield {"type": "response", "data": f"\n\n[System Error: {error_msg}]"}
                        self._record_error_as_final(error_msg)
                        return

            if is_final_answer:
                return