import abc
import asyncio
import logging
import importlib.resources
from typing import Dict, List, Any
from llm_inference_engine import InferenceEngine
from medds_agent.history import History, Step, UserStep, AgentStep, ObservationStep, SystemStep
from medds_agent.utils import serialize_python_state, deserialize_python_state, apply_prompt_template

logger = logging.getLogger(__name__)


class AgentMemory(abc.ABC): 
    def __init__(self): 
        """ Abstract base class for agent memory management. """ 
        self.system_step = None 
        self.system_prompt = ""

        # Load templates
        file_path_obj = importlib.resources.files('medds_agent.asset.prompt_templates').joinpath("UserStep_prompt_template.md")
        with importlib.resources.as_file(file_path_obj) as file_path:
            with open(file_path, "r") as file:
                self.user_step_template = file.read()
        
        file_path_obj = importlib.resources.files('medds_agent.asset.prompt_templates').joinpath("ObservationStep_toolcall_prompt_template.md")
        with importlib.resources.as_file(file_path_obj) as file_path:
            with open(file_path, "r") as file:
                self.tool_call_template = file.read()

        file_path_obj = importlib.resources.files('medds_agent.asset.prompt_templates').joinpath("AgentStep_toolcall_prompt_template.md")
        with importlib.resources.as_file(file_path_obj) as file_path:
            with open(file_path, "r") as file:
                self.agent_step_toolcall_template = file.read()

    def set_system_prompt(self, system_prompt:str, specialty_prompt:str=None) -> None:
        """
        Sets the system prompt for the memory.

        Parameters:
        ----------
        system_prompt : str
            The system prompt to set.
        specialty_prompt : str, optional
            Domain or task-specific instructions that can be appended to the system prompt for more context.
        """
        self.system_prompt = system_prompt
        if specialty_prompt:
            self.system_prompt += f"\n\n{specialty_prompt}"

    @abc.abstractmethod
    def get_messages(self, history: History) -> List[Dict[str, str]]:
        """
        Returns a list of OpenAI chat completion API messages.

        Parameters:
        ----------
        history : History
            The history object containing all steps.
        """
        return NotImplemented

    @abc.abstractmethod
    def on_step_added(self, step: Step) -> None:
        """
        Hook that is called when a new step is added to the memory (sync).

        Parameters:
        ----------
        step : Step
            The step that was added.
        """
        return NotImplemented

    @abc.abstractmethod
    async def on_step_added_async(self, step: Step) -> None:
        """
        Async hook that is called when a new step is added to the memory.

        Parameters:
        ----------
        step : Step
            The step that was added.
        """
        return NotImplemented

    @abc.abstractmethod
    def serialize(self) -> Dict[str, Any]:
        return NotImplemented

    @abc.abstractmethod
    def deserialize(self, data: Dict[str, Any]) -> None:
        return NotImplemented

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Resets the memory to its initial state.
        """
        return NotImplemented

    def _build_debug_entry(self, index, formatted_msg, step=None, round_num=None,
                           step_type="system_prompt", is_compressed=False,
                           original_content_length=None):
        """Helper to build a debug entry for memory inspection."""
        entry = {
            "index": index,
            "role": formatted_msg["role"],
            "content": formatted_msg["content"],
            "content_length": len(formatted_msg["content"]),
            "step_id": step.step_id if step else None,
            "round_num": round_num,
            "step_type": step_type,
            "is_compressed": is_compressed,
        }
        if original_content_length is not None:
            entry["original_content_length"] = original_content_length
        return entry

    def get_memory_debug(self, history: History) -> Dict[str, Any]:
        """Returns debug info about what messages the LLM receives. Override for detailed annotations."""
        messages = self.get_messages(history)
        debug_messages = []
        for i, msg in enumerate(messages):
            debug_messages.append({
                "index": i,
                "role": msg.get("role", ""),
                "content": msg.get("content", ""),
                "content_length": len(msg.get("content", "")),
                "step_id": None,
                "round_num": None,
                "step_type": "system_prompt" if i == 0 else "unknown",
                "is_compressed": False
            })
        return {
            "memory_type": self.__class__.__name__,
            "config": {},
            "summary": {
                "total_rounds": len(history.rounds),
                "total_messages": len(debug_messages),
                "total_content_length": sum(m["content_length"] for m in debug_messages)
            },
            "messages": debug_messages
        }

    def get_compression_debug(self, history: History) -> Dict[str, Any]:
        """Returns debug info about compressed content. Default: not supported."""
        return {
            "memory_type": self.__class__.__name__,
            "compression_supported": False,
            "entries": []
        }

    def _format_user_step(self, step: UserStep, system_steps: List[SystemStep]) -> Dict[str, str]:
        """
        Helper to format a UserStep into a message, aggregating preceding SystemSteps.
        """
        return {
            "role": "user",
            "content": apply_prompt_template(self.user_step_template, {
                "requested_time": step.start_time.strftime("%Y-%m-%d %H:%M:%S"), 
                "system_message": "\n".join([s.system_message for s in system_steps]) if system_steps else "No system messages.",
                "user_input": step.user_input
            }),
        }

    def _format_observation_step(self, step: ObservationStep) -> Dict[str, str]:
        """
        Helper to format a ObservationStep into a message.
        """
        if step.output:
            content = apply_prompt_template(self.tool_call_template, {"output": step.output})
        else:
            content = step.output
        return {
            "role": "user",
            "content": content,
        }
    
    def _format_agent_step(self, step: AgentStep) -> Dict[str, str]:
        if step.tool_name:
            return {
                "role": "assistant",
                "content": apply_prompt_template(self.agent_step_toolcall_template, {
                    "tool_name": step.tool_name,
                    "tool_args": step.tool_args
                }),
            }
        else:
            return {
                "role": "assistant",
                "content": step.response,
            }

    
class StateLessAgentMemory(AgentMemory): 
    def __init__(self): 
        """ An abstract base class for stateless memories. This is good for on-the-fly memory generation. """ 
        super().__init__()

    def reset(self) -> None:
        # No action needed
        pass

class FullHistoryAgentMemory(StateLessAgentMemory):
    def __init__(self):
        """ A full history memory that keeps all steps in memory. """
        super().__init__()

    def on_step_added(self, step: Step) -> None:
        pass

    async def on_step_added_async(self, step: Step) -> None:
        pass

    def get_messages(self, history: History) -> List[Dict[str, str]]:
        system_step_message = {"role": "system", "content": self.system_prompt}
        messages = [system_step_message]
        
        for i, round in enumerate(history.rounds):
            relevant_system_steps = []
            
            # Add system steps from Previous Round (if exists)
            if i > 0:
                prev_round = history.rounds[i-1]
                relevant_system_steps.extend([s for s in prev_round.steps if isinstance(s, SystemStep)])
            
            # Add system steps from Current Round
            relevant_system_steps.extend([s for s in round.steps if isinstance(s, SystemStep)])

            # Process steps in the current round
            for step in round.steps:
                if isinstance(step, UserStep):
                    # Attach the gathered system steps to the UserStep
                    messages.append(self._format_user_step(step, relevant_system_steps))
                
                # SystemSteps are not added as standalone messages
                elif isinstance(step, SystemStep):
                    continue
                
                elif isinstance(step, ObservationStep):
                    messages.append(self._format_observation_step(step))
                
                elif isinstance(step, AgentStep):
                    messages.append(self._format_agent_step(step))
        
        return messages

    def get_memory_debug(self, history: History) -> Dict[str, Any]:
        debug_messages = []
        msg_idx = 0

        system_msg = {"role": "system", "content": self.system_prompt}
        debug_messages.append(self._build_debug_entry(msg_idx, system_msg))
        msg_idx += 1

        for i, round in enumerate(history.rounds):
            relevant_system_steps = []
            if i > 0:
                prev_round = history.rounds[i - 1]
                relevant_system_steps.extend([s for s in prev_round.steps if isinstance(s, SystemStep)])
            relevant_system_steps.extend([s for s in round.steps if isinstance(s, SystemStep)])

            for step in round.steps:
                if isinstance(step, UserStep):
                    formatted = self._format_user_step(step, relevant_system_steps)
                    debug_messages.append(self._build_debug_entry(
                        msg_idx, formatted, step, round.round_num, "UserStep"))
                    msg_idx += 1
                elif isinstance(step, SystemStep):
                    continue
                elif isinstance(step, ObservationStep):
                    formatted = self._format_observation_step(step)
                    debug_messages.append(self._build_debug_entry(
                        msg_idx, formatted, step, round.round_num, "ObservationStep"))
                    msg_idx += 1
                elif isinstance(step, AgentStep):
                    formatted = self._format_agent_step(step)
                    debug_messages.append(self._build_debug_entry(
                        msg_idx, formatted, step, round.round_num, "AgentStep"))
                    msg_idx += 1

        total_content_length = sum(m["content_length"] for m in debug_messages)
        return {
            "memory_type": "FullHistoryAgentMemory",
            "config": {},
            "summary": {
                "total_rounds": len(history.rounds),
                "total_messages": len(debug_messages),
                "total_content_length": total_content_length
            },
            "messages": debug_messages
        }

    def serialize(self) -> Dict[str, Any]:
        return {}

    def deserialize(self, data: Dict[str, Any]) -> None:
        pass

class SlideWindowAgentMemory(StateLessAgentMemory):
    def __init__(self, start_window_size: int = 5, end_window_size: int = 20): 
        """ 
        A sliding window memory that keeps full steps for the first N and last M rounds in memory. For rounds in the middle, only the user and answer steps are kept, while discarding coding and code output steps.

        Parameters:
        ----------
        start_window_size : int
            The number of initial rounds to keep in memory.
        end_window_size : int
            The number of most recent rounds to keep in memory.
        """
        super().__init__()
        self.start_window_size = start_window_size
        self.end_window_size = end_window_size

    def get_messages(self, history: History) -> List[Dict[str, str]]:
        system_step_message = {"role": "system", "content": self.system_prompt}
        messages = [system_step_message]
        
        total_rounds = len(history.rounds)

        for i, round in enumerate(history.rounds):
            is_full_round = (round.round_num <= self.start_window_size) or \
                            (round.round_num > total_rounds - self.end_window_size)
            
            relevant_system_steps = []
            # Add system steps from Previous Round (if exists)
            if i > 0:
                prev_round = history.rounds[i-1]
                relevant_system_steps.extend([s for s in prev_round.steps if isinstance(s, SystemStep)])
            
            # Add system steps from Current Round
            relevant_system_steps.extend([s for s in round.steps if isinstance(s, SystemStep)])

            # Process steps in the current round
            for step in round.steps:
                if isinstance(step, UserStep):
                    # User step is always kept and gets the system context
                    messages.append(self._format_user_step(step, relevant_system_steps))
                
                elif is_full_round:
                    # For full rounds, keep all other steps
                    if isinstance(step, ObservationStep):
                        messages.append(self._format_observation_step(step))
                    elif isinstance(step, AgentStep):
                        messages.append(self._format_agent_step(step))
                
                else:
                    # For middle rounds, only keep last AgentStep
                    if isinstance(step, AgentStep) and step.is_final:
                        messages.append(self._format_agent_step(step))

        return messages

    def get_memory_debug(self, history: History) -> Dict[str, Any]:
        debug_messages = []
        msg_idx = 0
        total_rounds = len(history.rounds)
        full_rounds = []
        windowed_rounds = []

        system_msg = {"role": "system", "content": self.system_prompt}
        debug_messages.append(self._build_debug_entry(msg_idx, system_msg))
        msg_idx += 1

        for i, round in enumerate(history.rounds):
            is_full_round = (round.round_num <= self.start_window_size) or \
                            (round.round_num > total_rounds - self.end_window_size)
            if is_full_round:
                full_rounds.append(round.round_num)
            else:
                windowed_rounds.append(round.round_num)

            relevant_system_steps = []
            if i > 0:
                prev_round = history.rounds[i - 1]
                relevant_system_steps.extend([s for s in prev_round.steps if isinstance(s, SystemStep)])
            relevant_system_steps.extend([s for s in round.steps if isinstance(s, SystemStep)])

            for step in round.steps:
                if isinstance(step, UserStep):
                    formatted = self._format_user_step(step, relevant_system_steps)
                    debug_messages.append(self._build_debug_entry(
                        msg_idx, formatted, step, round.round_num, "UserStep"))
                    msg_idx += 1
                elif is_full_round:
                    if isinstance(step, ObservationStep):
                        formatted = self._format_observation_step(step)
                        debug_messages.append(self._build_debug_entry(
                            msg_idx, formatted, step, round.round_num, "ObservationStep"))
                        msg_idx += 1
                    elif isinstance(step, AgentStep):
                        formatted = self._format_agent_step(step)
                        debug_messages.append(self._build_debug_entry(
                            msg_idx, formatted, step, round.round_num, "AgentStep"))
                        msg_idx += 1
                else:
                    if isinstance(step, AgentStep) and step.is_final:
                        formatted = self._format_agent_step(step)
                        debug_messages.append(self._build_debug_entry(
                            msg_idx, formatted, step, round.round_num, "AgentStep"))
                        msg_idx += 1

        total_content_length = sum(m["content_length"] for m in debug_messages)
        return {
            "memory_type": "SlideWindowAgentMemory",
            "config": {
                "start_window_size": self.start_window_size,
                "end_window_size": self.end_window_size
            },
            "summary": {
                "total_rounds": total_rounds,
                "full_rounds": full_rounds,
                "windowed_rounds": windowed_rounds,
                "total_messages": len(debug_messages),
                "total_content_length": total_content_length
            },
            "messages": debug_messages
        }

    def on_step_added(self, step: Step) -> None:
        pass

    async def on_step_added_async(self, step: Step) -> None:
        pass

    def serialize(self) -> Dict[str, Any]:
        return {
            "start_window_size": self.start_window_size,
            "end_window_size": self.end_window_size
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        self.start_window_size = data["start_window_size"]
        self.end_window_size = data["end_window_size"]

class StatedAgentMemory(AgentMemory):
    def __init__(self): 
        """ An abstract base class for stateful memories. This is good for persistent memory storage. """ 
        super().__init__()

    def serialize(self) -> Dict[str, Any]:
        return {
            "python_state": serialize_python_state(self.python_state)
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        self.python_state = deserialize_python_state(data["python_state"])


class IndexedAgentMemory(StatedAgentMemory):
    COMPRESSED_PREFIX = "[COMPRESSED MESSAGE]\n"

    def __init__(self, llm_engine: InferenceEngine, compress_threshold: int = 2000,
                 recent_window_size: int = 5, start_window_size: int = 3):
        """
        A stateful memory that compresses older step content via LLM summarization.

        When a step is added whose formatted content exceeds `compress_threshold` characters,
        the content is sent to the LLM for compression. During `get_messages`, steps in the
        most recent `recent_window_size` rounds and first `start_window_size` rounds use full
        content; middle rounds use compressed content when available.

        Parameters:
        ----------
        llm_engine : InferenceEngine
            The LLM inference engine used for compressing step content.
        compress_threshold : int
            Character count threshold above which step content is compressed.
        recent_window_size : int
            Number of most recent rounds that use full (uncompressed) content.
        start_window_size : int
            Number of initial rounds that use full (uncompressed) content.
        """
        super().__init__()
        self._llm_engine = llm_engine
        self._compress_threshold = compress_threshold
        self._recent_window_size = recent_window_size
        self._start_window_size = start_window_size
        self._compressed_cache: Dict[str, str] = {}
        self._pending_tasks: set = set()

        # Load compression prompt template
        file_path_obj = importlib.resources.files('medds_agent.asset.prompt_templates').joinpath(
            "IndexedMemory_compress_prompt_template.md")
        with importlib.resources.as_file(file_path_obj) as file_path:
            with open(file_path, "r") as file:
                self._compress_template = file.read()

    def _get_step_type_label(self, step: Step) -> str:
        """Return a human-readable type label for the compression prompt."""
        if isinstance(step, UserStep):
            return "user input"
        elif isinstance(step, AgentStep):
            return "agent response"
        elif isinstance(step, ObservationStep):
            return "tool observation"
        elif isinstance(step, SystemStep):
            return "system message"
        return "unknown"

    def _get_formatted_content(self, step: Step) -> str:
        """Get the formatted content string for a step (same as what goes into messages)."""
        if isinstance(step, UserStep):
            return self._format_user_step(step, [])["content"]
        elif isinstance(step, ObservationStep):
            return self._format_observation_step(step)["content"]
        elif isinstance(step, AgentStep):
            return self._format_agent_step(step)["content"]
        elif isinstance(step, SystemStep):
            return step.system_message
        return ""

    def _compress_sync(self, step: Step, content: str) -> None:
        """Synchronously compress a step's content and cache the result."""
        type_label = self._get_step_type_label(step)
        prompt = apply_prompt_template(self._compress_template, {
            "type": type_label,
            "content": content
        })
        messages = [{"role": "user", "content": prompt}]
        response = self._llm_engine.chat(messages=messages, tools=[])
        compressed = response.get("response", content)
        self._compressed_cache[step.step_id] = self.COMPRESSED_PREFIX + compressed

    async def _compress_async(self, step: Step, content: str) -> None:
        """Asynchronously compress a step's content and cache the result."""
        type_label = self._get_step_type_label(step)
        prompt = apply_prompt_template(self._compress_template, {
            "type": type_label,
            "content": content
        })
        messages = [{"role": "user", "content": prompt}]
        response = await self._llm_engine.chat_async(messages=messages, tools=[])
        compressed = response.get("response", content)
        self._compressed_cache[step.step_id] = self.COMPRESSED_PREFIX + compressed

    def on_step_added(self, step: Step) -> None:
        """Sync hook: compress step content if it exceeds the threshold (blocking)."""
        content = self._get_formatted_content(step)
        if len(content) > self._compress_threshold:
            try:
                self._compress_sync(step, content)
            except Exception as e:
                logger.warning(f"Failed to compress step {step.step_id}: {e}")

    async def on_step_added_async(self, step: Step) -> None:
        """Async hook: fire-and-forget compression task if content exceeds the threshold."""
        content = self._get_formatted_content(step)
        if len(content) > self._compress_threshold:
            task = asyncio.create_task(self._compress_async(step, content))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

    def get_messages(self, history: History) -> List[Dict[str, str]]:
        system_step_message = {"role": "system", "content": self.system_prompt}
        messages = [system_step_message]

        total_rounds = len(history.rounds)

        for i, round in enumerate(history.rounds):
            is_full_round = (round.round_num <= self._start_window_size) or \
                            (round.round_num > total_rounds - self._recent_window_size)

            relevant_system_steps = []
            if i > 0:
                prev_round = history.rounds[i - 1]
                relevant_system_steps.extend([s for s in prev_round.steps if isinstance(s, SystemStep)])
            relevant_system_steps.extend([s for s in round.steps if isinstance(s, SystemStep)])

            for step in round.steps:
                if isinstance(step, UserStep):
                    messages.append(self._format_user_step(step, relevant_system_steps))

                elif isinstance(step, SystemStep):
                    continue

                elif is_full_round or step.step_id not in self._compressed_cache:
                    # Full round or no compressed version available: use original content
                    if isinstance(step, ObservationStep):
                        messages.append(self._format_observation_step(step))
                    elif isinstance(step, AgentStep):
                        messages.append(self._format_agent_step(step))

                else:
                    # Middle round with compressed content available
                    compressed_content = self._compressed_cache[step.step_id]
                    if isinstance(step, ObservationStep):
                        messages.append(self._format_observation_step(
                            ObservationStep(
                                agent_id=step.agent_id,
                                start_time=step.start_time,
                                end_time=step.end_time,
                                output=compressed_content,
                                step_id=step.step_id
                            )
                        ))
                    elif isinstance(step, AgentStep):
                        if step.tool_name:
                            messages.append({
                                "role": "assistant",
                                "content": apply_prompt_template(self.agent_step_toolcall_template, {
                                    "tool_name": step.tool_name,
                                    "tool_args": compressed_content
                                }),
                            })
                        else:
                            messages.append({
                                "role": "assistant",
                                "content": compressed_content,
                            })

        return messages

    def serialize(self) -> Dict[str, Any]:
        return {
            "compressed_cache": self._compressed_cache,
            "compress_threshold": self._compress_threshold,
            "recent_window_size": self._recent_window_size,
            "start_window_size": self._start_window_size,
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        self._compressed_cache = data.get("compressed_cache", {})
        self._compress_threshold = data.get("compress_threshold", self._compress_threshold)
        self._recent_window_size = data.get("recent_window_size", self._recent_window_size)
        self._start_window_size = data.get("start_window_size", self._start_window_size)

    def reset(self) -> None:
        self._compressed_cache.clear()
        for task in self._pending_tasks:
            task.cancel()
        self._pending_tasks.clear()

    def get_memory_debug(self, history: History) -> Dict[str, Any]:
        debug_messages = []
        msg_idx = 0
        total_rounds = len(history.rounds)
        compressed_steps_count = 0
        full_rounds = []
        compressed_rounds = []

        system_msg = {"role": "system", "content": self.system_prompt}
        debug_messages.append(self._build_debug_entry(msg_idx, system_msg))
        msg_idx += 1

        for i, round in enumerate(history.rounds):
            is_full_round = (round.round_num <= self._start_window_size) or \
                            (round.round_num > total_rounds - self._recent_window_size)
            if is_full_round:
                full_rounds.append(round.round_num)
            else:
                compressed_rounds.append(round.round_num)

            relevant_system_steps = []
            if i > 0:
                prev_round = history.rounds[i - 1]
                relevant_system_steps.extend([s for s in prev_round.steps if isinstance(s, SystemStep)])
            relevant_system_steps.extend([s for s in round.steps if isinstance(s, SystemStep)])

            for step in round.steps:
                if isinstance(step, UserStep):
                    formatted = self._format_user_step(step, relevant_system_steps)
                    debug_messages.append(self._build_debug_entry(
                        msg_idx, formatted, step, round.round_num, "UserStep"))
                    msg_idx += 1

                elif isinstance(step, SystemStep):
                    continue

                elif is_full_round or step.step_id not in self._compressed_cache:
                    if isinstance(step, ObservationStep):
                        formatted = self._format_observation_step(step)
                        debug_messages.append(self._build_debug_entry(
                            msg_idx, formatted, step, round.round_num, "ObservationStep"))
                        msg_idx += 1
                    elif isinstance(step, AgentStep):
                        formatted = self._format_agent_step(step)
                        debug_messages.append(self._build_debug_entry(
                            msg_idx, formatted, step, round.round_num, "AgentStep"))
                        msg_idx += 1

                else:
                    compressed_content = self._compressed_cache[step.step_id]
                    compressed_steps_count += 1

                    if isinstance(step, ObservationStep):
                        original_formatted = self._format_observation_step(step)
                        compressed_obs = ObservationStep(
                            agent_id=step.agent_id,
                            start_time=step.start_time,
                            end_time=step.end_time,
                            output=compressed_content,
                            step_id=step.step_id
                        )
                        formatted = self._format_observation_step(compressed_obs)
                        debug_messages.append(self._build_debug_entry(
                            msg_idx, formatted, step, round.round_num, "ObservationStep",
                            is_compressed=True,
                            original_content_length=len(original_formatted["content"])))
                        msg_idx += 1
                    elif isinstance(step, AgentStep):
                        original_formatted = self._format_agent_step(step)
                        if step.tool_name:
                            formatted = {
                                "role": "assistant",
                                "content": apply_prompt_template(self.agent_step_toolcall_template, {
                                    "tool_name": step.tool_name,
                                    "tool_args": compressed_content
                                }),
                            }
                        else:
                            formatted = {
                                "role": "assistant",
                                "content": compressed_content,
                            }
                        debug_messages.append(self._build_debug_entry(
                            msg_idx, formatted, step, round.round_num, "AgentStep",
                            is_compressed=True,
                            original_content_length=len(original_formatted["content"])))
                        msg_idx += 1

        total_content_length = sum(m["content_length"] for m in debug_messages)
        return {
            "memory_type": "IndexedAgentMemory",
            "config": {
                "start_window_size": self._start_window_size,
                "recent_window_size": self._recent_window_size,
                "compress_threshold": self._compress_threshold
            },
            "summary": {
                "total_rounds": total_rounds,
                "full_rounds": full_rounds,
                "compressed_rounds": compressed_rounds,
                "total_messages": len(debug_messages),
                "compressed_steps": compressed_steps_count,
                "pending_compressions": len(self._pending_tasks),
                "total_content_length": total_content_length
            },
            "messages": debug_messages
        }

    def get_compression_debug(self, history: History) -> Dict[str, Any]:
        entries = []
        for rnd in history.rounds:
            for step in rnd.steps:
                if step.step_id in self._compressed_cache:
                    original_content = self._get_formatted_content(step)
                    compressed_content = self._compressed_cache[step.step_id]
                    original_len = len(original_content)
                    compressed_len = len(compressed_content)
                    entries.append({
                        "step_id": step.step_id,
                        "step_type": step.__class__.__name__,
                        "round_num": rnd.round_num,
                        "original_content": original_content,
                        "compressed_content": compressed_content,
                        "original_length": original_len,
                        "compressed_length": compressed_len,
                        "compression_ratio": round(compressed_len / original_len, 4) if original_len > 0 else None
                    })
        return {
            "memory_type": "IndexedAgentMemory",
            "compression_supported": True,
            "config": {
                "compress_threshold": self._compress_threshold
            },
            "total_compressed": len(self._compressed_cache),
            "pending_compressions": len(self._pending_tasks),
            "entries": entries
        }