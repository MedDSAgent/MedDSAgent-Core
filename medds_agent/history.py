import abc
import uuid
from typing import Union, Dict, List, Any
from datetime import datetime


class Step(abc.ABC):
    """
    Steps are the minimal unit of History. Each step represents a 1) system event, 2) user input, 3) tool call, or 4) assistant answer.
    Steps do not belong to any agent. They are simply records of what happened at a specific time. 
    """
    def __init__(self, start_time: datetime, end_time: datetime, step_id: str=None):
        """
        Abstract step class.

        Parameters:
        -----------
        start_time: datetime
            The start time of the step.
        end_time: datetime
            The end time of the step.
        step_id: str, Optional
            The unique identifier for the step. If not provided, a new UUID will be generated.
        """
        self.step_id = step_id if step_id else str(uuid.uuid4())
        self.start_time = start_time
        self.end_time = end_time
    
    @abc.abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the step to a dictionary.
        """
        return NotImplemented
    
    @classmethod
    @abc.abstractmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'Step':
        """
        Deserialize the step from a dictionary.
        """
        return NotImplemented
    

class SystemStep(Step):
    def __init__(self, start_time: datetime, end_time: datetime, system_message: str, step_id: str=None):
        """
        System message step, such as user interruptions, file uploads, etc. Does NOT include system prompt!

        Parameters:
        -----------
        start_time: datetime
            The start time of the step.
        end_time: datetime
            The end time of the step.
        system_message: str
            The system message content.
        step_id: str, Optional
            The unique identifier for the step.
        """
        super().__init__(start_time, end_time, step_id=step_id)
        self.system_message = system_message

    def __repr__(self):
        return (f"SystemStep(system_message={self.system_message}, "
                f"start_time={self.start_time}, "
                f"end_time={self.end_time})")
    
    def serialize(self) -> Dict[str, Any]:
        return {
            "type": "SystemStep",
            "step_id": self.step_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "system_message": self.system_message
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'SystemStep':
        return cls(
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            system_message=data["system_message"],
            step_id=data.get("step_id")
        )


class UserStep(Step):
    def __init__(self, start_time: datetime, end_time: datetime, user_input:str, step_id: str=None):
        """
        User input step.

        Parameters:
        -----------
        start_time: datetime
            The start time of the step.
        end_time: datetime
            The end time of the step.
        user_input: str
            The user's input.
        step_id: str, Optional
            The unique identifier for the step.
        """
        super().__init__(start_time, end_time, step_id=step_id)
        self.user_input = user_input

    def __repr__(self):
        return (f"UserStep(start_time={self.start_time}, end_time={self.end_time}, user_input={self.user_input})")
    
    def serialize(self):
        return {
            "type": "UserStep",
            "step_id": self.step_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "user_input": self.user_input
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'UserStep':
        return cls(
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            user_input=data["user_input"],
            step_id=data.get("step_id")
        )
    

class AgentStep(Step):
    def __init__(self, agent_id: str, start_time: datetime, end_time: datetime, response: str=None, tool_name: str=None, tool_args: str=None, is_final: bool=False, step_id: str=None):
        """
        Assistant action step.

        Parameters:
        -----------
        agent_id: str
            The unique identifier for the agent.
        start_time: datetime
            The start time of the step.
        end_time: datetime
            The end time of the step.
        response: str, Optional
            The assistant's response.
        tool_name: str, Optional
            The name of the tool to call.
        tool_args: str, Optional
            The serialized arguments.
        is_final: bool, Optional
            Whether this step is the final answer.
        step_id: str, Optional
            The unique identifier for the step.
        """
        super().__init__(start_time, end_time, step_id=step_id)
        self.agent_id = agent_id
        self.response = response
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.is_final = is_final

    def __repr__(self):
        return (f"AgentStep(agent_id={self.agent_id}, start_time={self.start_time}, end_time={self.end_time}, response={self.response}, tool_name={self.tool_name}, tool_args={self.tool_args}, is_final={self.is_final})")
        
    def serialize(self) -> Dict[str, Any]:
        return {
            "type": "AgentStep",
            "step_id": self.step_id,
            "agent_id": self.agent_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "response": self.response,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "is_final": self.is_final
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'AgentStep':
        return cls(
            agent_id=data["agent_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            response=data["response"],
            tool_name=data["tool_name"],
            tool_args=data["tool_args"],
            is_final=data.get("is_final", False),
            step_id=data.get("step_id")
        )
    

class ObservationStep(Step):
    def __init__(self, agent_id: str, start_time: datetime, end_time: datetime, output:str, step_id: str=None):
        """
        Tool output step.

        Parameters:
        -----------
        agent_id: str
            The unique identifier for the agent.
        start_time: datetime
            The start time of the step.
        end_time: datetime
            The end time of the step.
        output: str
            The output from the tool that was executed.
        step_id: str, Optional
            The unique identifier for the step.
        """
        super().__init__(start_time, end_time, step_id=step_id)
        self.agent_id = agent_id
        self.output = output

    def __repr__(self):
        return (f"ObservationStep(agent_id={self.agent_id}, start_time={self.start_time}, end_time={self.end_time}, output={self.output})")
    
    def serialize(self) -> Dict[str, Any]:
        return {
            "type": "ObservationStep",
            "step_id": self.step_id,
            "agent_id": self.agent_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "output": self.output
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'ObservationStep':
        return cls(
            agent_id=data["agent_id"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            output=data["output"],
            step_id=data.get("step_id")
        )
      

class Round:
    def __init__(self, round_num:int):
        """
        A round of conversation consisting of multiple steps. Start with a UserStep and ends with an AgentStep.

        Parameters:
        -----------
        round_num: int
            The round number.
        """
        self.round_num = round_num
        self.steps = []

    def __repr__(self):
        return (f"Round(round_num={self.round_num}, steps={self.steps})")
    
    def __len__(self):
        return len(self.steps)

    def add_step(self, step: Step):
        """
        Add a step to the round.

        Parameters:
        -----------
        step: Step
            The step to add.
        """
        self.steps.append(step)

    def get_user_step(self) -> UserStep:
        """
        Get the UserStep of the round.

        Returns:
        -----------
        UserStep
            The UserStep of the round.
        """
        for step in self.steps:
            if isinstance(step, UserStep):
                return step
        raise ValueError("No UserStep found in the round.")
    
    def get_answer_step(self) -> AgentStep:
        """
        Get the AssistantStep with answer of the round.

        Returns:
        -----------
        AgentStep
            The AgentStep with answer of the round.
        """
        for step in self.steps:
            if isinstance(step, AgentStep):
                return step
        raise ValueError("No AgentStep with answer found in the round.")

    def serialize(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "steps": [step.serialize() for step in self.steps]
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'Round':
        round_instance = cls(round_num=data["round_num"])
        for step_data in data["steps"]:
            step_type = step_data.get("type")
            if step_type == "SystemStep":
                step = SystemStep.deserialize(step_data)
            elif step_type == "UserStep":
                step = UserStep.deserialize(step_data)
            elif step_type == "ObservationStep":
                step = ObservationStep.deserialize(step_data)
            elif step_type == "AgentStep":
                step = AgentStep.deserialize(step_data)
            else:
                raise ValueError(f"Unknown step type: {step_type}")
            round_instance.add_step(step)
        return round_instance


class History:
    def __init__(self):
        """
        History of the conversation. Consists of a system step and multiple rounds (UserStep -> AgentStep -> ObservationStep -> AgentStep).
        """
        self.rounds = []
        self.current_round_num = 0

    def reset(self) -> None:
        """
        Resets the history to its initial state.
        """
        self.rounds = []
        self.current_round_num = 0

    def __repr__(self):
        return (f"History(num_rounds={len(self.rounds)}, rounds={self.rounds[0:2]})\n...")

    def add_step(self, step: Step):
        """
        Add a step to the history. If the step is a UserStep, start a new round.

        Parameters:
        -----------
        step: Step
            The step to add.
        """
        # User step starts a new round
        if isinstance(step, UserStep):
            self.current_round_num += 1
            new_round = Round(self.current_round_num)
            new_round.add_step(step)
            self.rounds.append(new_round)

        # Other steps are added to the current round
        else:
            if not self.rounds:
                # If the first step is a SystemStep, create a new round for it
                if isinstance(step, SystemStep):
                    new_round = Round(0)
                    new_round.add_step(step)
                    self.rounds.append(new_round)
                    return
                else:
                    raise ValueError("Cannot add non-UserStep before any UserStep.")
            self.rounds[-1].add_step(step)
    
    def serialize(self) -> Dict[str, Any]:
        return {
            "rounds": [round.serialize() for round in self.rounds]
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'History':
        history = cls()
        for round_data in data["rounds"]:
            round_instance = Round.deserialize(round_data)
            history.rounds.append(round_instance)
        history.current_round_num = len(history.rounds)
        return history