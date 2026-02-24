"""
MedDSAgent — Medical Data Science Agent backend library.

Core public API:
    Agent           — the main action-observation loop agent
    History         — conversation history manager
    AgentMemory     — base class for memory strategies
    Tool            — base class for agent tools

Convenience re-exports of the most-used classes are provided below.
The full API is available from the sub-modules directly.
"""

from medds_agent.agents import Agent
from medds_agent.history import History, UserStep, AgentStep, ObservationStep, SystemStep
from medds_agent.memory import (
    AgentMemory,
    FullHistoryAgentMemory,
    SlideWindowAgentMemory,
    IndexedAgentMemory,
)
from medds_agent.tools import Tool, PythonExecutorTool, RExecutorTool, FileSystemTool, DocumentSearchTool
from medds_agent.engines import (
    AgentBasicLLMConfig,
    AgentReasoningLLMConfig,
    AgentQwen3LLMConfig,
    AgentOpenAIReasoningLLMConfig,
)

__version__ = "0.1.0"

__all__ = [
    # Agent
    "Agent",
    # History
    "History",
    "UserStep",
    "AgentStep",
    "ObservationStep",
    "SystemStep",
    # Memory
    "AgentMemory",
    "FullHistoryAgentMemory",
    "SlideWindowAgentMemory",
    "IndexedAgentMemory",
    # Tools
    "Tool",
    "PythonExecutorTool",
    "RExecutorTool",
    "FileSystemTool",
    "DocumentSearchTool",
    # LLM configs
    "AgentBasicLLMConfig",
    "AgentReasoningLLMConfig",
    "AgentQwen3LLMConfig",
    "AgentOpenAIReasoningLLMConfig",
]