"""
Command-line interface for the Medical Data Science Agent.

This CLI is intended for local debugging and internal testing.
For production use, start the full server with: medds-server

Note: DocumentSearchTool is not available here — it requires a database-backed
session managed by SessionManager, which is only active when the server runs.

Usage:
    # Interactive mode
    python -m medds_agent.cli

    # Single query and exit
    python -m medds_agent.cli -q "What files are available?"

    # Custom workspace with a different model
    python -m medds_agent.cli -w ./my_data -p openai -m gpt-4.1

    # R language mode
    python -m medds_agent.cli --language r
"""
import os
import sys
import argparse
import readline
import dill
from datetime import datetime
from typing import Optional, List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from medds_agent.agents import Agent
from medds_agent.history import History, SystemStep
from medds_agent.memory import SlideWindowAgentMemory
from medds_agent.tools import Tool, PythonExecutorTool, RExecutorTool, FileSystemTool
from medds_agent.engines import (
    OpenAIInferenceEngine,
    AzureOpenAIInferenceEngine,
    VLLMInferenceEngine,
    SGLangInferenceEngine,
    OpenRouterInferenceEngine,
    AgentBasicLLMConfig,
    AgentReasoningLLMConfig,
)

_STATE_FILE = "state.pkl"


# =============================================================================
# Workspace setup
# =============================================================================

def _setup_work_dir(work_dir: str) -> str:
    """Create the workspace and all required subdirectories."""
    work_dir = os.path.abspath(work_dir)
    for subdir in ("uploads", "outputs", "scripts", "internal"):
        os.makedirs(os.path.join(work_dir, subdir), exist_ok=True)
    return work_dir


# =============================================================================
# State persistence (Python only)
# =============================================================================

def save_state(python_tool: PythonExecutorTool, work_dir: str) -> None:
    """Persist Python globals to state.pkl, excluding unpicklable DB objects."""
    state_path = os.path.join(work_dir, _STATE_FILE)
    state_to_save = {
        k: v for k, v in python_tool._globals.items()
        if k not in ("db_engine", "conn")
    }
    try:
        with open(state_path, "wb") as f:
            dill.dump(state_to_save, f)
    except Exception as e:
        print(f"[warn] Could not save Python state: {e}")


def load_state(python_tool: PythonExecutorTool, work_dir: str) -> None:
    """Load persisted Python state into the tool's globals if state.pkl exists."""
    state_path = os.path.join(work_dir, _STATE_FILE)
    if not os.path.exists(state_path):
        return
    try:
        with open(state_path, "rb") as f:
            python_tool._globals.update(dill.load(f))
        print(f"[info] Resumed Python state from {state_path}")
    except Exception as e:
        print(f"[warn] Could not load Python state: {e}")


# =============================================================================
# Tool factory
# =============================================================================

def _build_tools(
    work_dir: str,
    language: str = "python",
    db_connection_code: Optional[str] = None,
) -> Tuple[List[Tool], Optional[PythonExecutorTool], Optional[RExecutorTool]]:
    """
    Build the tool list for the CLI session.

    Returns (tools, python_tool, r_tool).
    """
    python_tool: Optional[PythonExecutorTool] = None
    r_tool: Optional[RExecutorTool] = None

    if language == "r":
        r_tool = RExecutorTool(work_dir=work_dir)
        executor: Tool = r_tool
        if db_connection_code and db_connection_code.strip():
            output = r_tool.execute({"code": db_connection_code})
            if "[Error]" in output:
                print(f"[warn] R DB connection code failed:\n{output}")
            else:
                print("[info] R DB connection code executed successfully.")
    else:
        python_tool = PythonExecutorTool(work_dir=work_dir)
        executor = python_tool
        if db_connection_code and db_connection_code.strip():
            try:
                exec(db_connection_code, python_tool._globals, python_tool._locals)
                python_tool._globals.update(python_tool._locals)
                print("[info] Database connection code executed successfully.")
            except Exception as e:
                print(f"[warn] DB connection code failed: {type(e).__name__}: {e}")

    tools: List[Tool] = [
        executor,
        FileSystemTool(dir=work_dir),
    ]
    return tools, python_tool, r_tool


# =============================================================================
# Agent factory
# =============================================================================

def create_agent(
    work_dir: str,
    provider: str = "openai",
    model: str = "gpt-4.1-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    max_new_tokens: int = 16384,
    max_input_tokens: int = 120000,
    temperature: float = 1.0,
    top_p: float = 1.0,
    language: str = "python",
    specialty_prompt: Optional[str] = None,
    db_connection_code: Optional[str] = None,
    verbose: bool = True,
    resume_state: bool = True,
) -> Tuple[Agent, Optional[PythonExecutorTool], Optional[RExecutorTool]]:
    """
    Build a configured Agent for CLI use.

    Returns (agent, python_tool, r_tool) so the REPL can save/inspect state.
    """
    work_dir = _setup_work_dir(work_dir)

    # --- LLM config ---
    if reasoning_effort and reasoning_effort in ("low", "medium", "high"):
        llm_config = AgentReasoningLLMConfig(
            max_input_tokens=max_input_tokens,
        )
    else:
        llm_config = AgentBasicLLMConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            max_input_tokens=max_input_tokens,
        )

    # --- Engine kwargs ---
    engine_kwargs = {"model": model, "config": llm_config}
    if api_key:
        engine_kwargs["api_key"] = api_key
        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
        elif provider == "azure":
            os.environ["AZURE_OPENAI_API_KEY"] = api_key
    if base_url:
        engine_kwargs["azure_endpoint" if provider == "azure" else "base_url"] = base_url

    # --- Engine ---
    if provider == "openai":
        llm_engine = OpenAIInferenceEngine(**engine_kwargs)
    elif provider == "azure":
        engine_kwargs["api_version"] = api_version or "2023-12-01-preview"
        llm_engine = AzureOpenAIInferenceEngine(**engine_kwargs)
    elif provider == "vllm":
        llm_engine = VLLMInferenceEngine(**engine_kwargs)
    elif provider == "sglang":
        llm_engine = SGLangInferenceEngine(**engine_kwargs)
    elif provider == "openrouter":
        llm_engine = OpenRouterInferenceEngine(**engine_kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    # --- Tools ---
    tools, python_tool, r_tool = _build_tools(work_dir, language, db_connection_code)

    # --- Resume Python state ---
    if resume_state and python_tool:
        load_state(python_tool, work_dir)

    # --- Agent ---
    agent = Agent(
        llm_engine=llm_engine,
        history=History(),
        agent_memory=SlideWindowAgentMemory(),
        specialty_prompt=specialty_prompt,
        tools=tools,
        verbose=verbose,
    )

    # --- System step for DB connection ---
    if db_connection_code and db_connection_code.strip() and python_tool:
        db_engine = python_tool._globals.get("db_engine")
        conn = python_tool._globals.get("conn")
        if db_engine is not None or conn is not None:
            var = "db_engine" if db_engine is not None else "conn"
            system_step = SystemStep(
                start_time=datetime.now(),
                end_time=datetime.now(),
                system_message=(
                    f"Database connection established. "
                    f"Use variable '{var}' to query the database."
                ),
            )
            agent.history.add_step(system_step)
            agent.agent_memory.on_step_added(system_step)

    return agent, python_tool, r_tool


# =============================================================================
# UI helpers
# =============================================================================

def _print_welcome(console: Console, work_dir: str, model: str, language: str) -> None:
    console.print()
    console.print(Panel.fit(
        "[bold blue]MedDSAgent[/bold blue]\n"
        "Medical Data Science Agent — debug CLI",
        border_style="blue"
    ))
    console.print()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")
    table.add_row("Work Directory", work_dir)
    table.add_row("Model", model)
    table.add_row("Language", language)
    console.print(table)
    console.print()
    console.print("[dim]Commands: exit · reset · state · history[/dim]")
    console.print()


def _print_state(console: Console, tool: Optional[Tool]) -> None:
    """Print the variable state from the active executor tool."""
    if tool is None:
        console.print("[dim]No executor tool active.[/dim]")
        return

    state = tool.get_state()
    if not state:
        console.print("[dim]No variables defined.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Value")

    for entry in state:
        name = f"[red]{entry['name']}[/red]" if entry.get("is_error") else entry["name"]
        table.add_row(name, entry.get("type", ""), entry.get("value", ""))

    console.print(table)


# =============================================================================
# Interactive REPL
# =============================================================================

def interactive_mode(
    agent: Agent,
    console: Console,
    work_dir: str,
    python_tool: Optional[PythonExecutorTool],
    r_tool: Optional[RExecutorTool],
) -> None:
    executor_tool: Optional[Tool] = python_tool or r_tool

    while True:
        try:
            try:
                user_input = input("\033[1;32mYou\033[0m: ")
            except EOFError:
                break

            cmd = user_input.lower().strip()

            if cmd == "exit":
                console.print("[yellow]Goodbye![/yellow]")
                break

            elif cmd == "reset":
                agent.history = History()
                if executor_tool and hasattr(executor_tool, "reset_state"):
                    executor_tool.reset_state()
                state_path = os.path.join(work_dir, _STATE_FILE)
                if os.path.exists(state_path):
                    os.remove(state_path)
                console.print("[yellow]History and state cleared.[/yellow]")
                continue

            elif cmd == "state":
                _print_state(console, executor_tool)
                continue

            elif cmd == "history":
                console.print(
                    f"[bold]Conversation History:[/bold] {len(agent.history.rounds)} rounds"
                )
                for r in agent.history.rounds[-5:]:
                    console.print(f"  Round {r.round_num}: {len(r.steps)} steps")
                continue

            elif not user_input.strip():
                continue

            # Run agent, then persist Python state
            agent.run(user_input)
            if python_tool:
                save_state(python_tool, work_dir)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MedDSAgent debug CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m medds_agent.cli
  python -m medds_agent.cli -q "Load data.csv and show summary"
  python -m medds_agent.cli -w ./my_data -p openai -m gpt-4.1
  python -m medds_agent.cli --language r
        """
    )

    parser.add_argument("--work-dir", "-w",
        default=os.environ.get("WORK_DIR", "./workspace"),
        help="Workspace directory (default: ./workspace or $WORK_DIR)")
    parser.add_argument("--provider", "-p",
        default=os.environ.get("LLM_PROVIDER", "openai"),
        help="LLM provider: openai|azure|vllm|sglang|openrouter (default: $LLM_PROVIDER)")
    parser.add_argument("--model", "-m",
        default=os.environ.get("LLM_MODEL", "gpt-4.1-mini"),
        help="LLM model name (default: $LLM_MODEL or gpt-4.1-mini)")
    parser.add_argument("--api-key",
        default=os.environ.get("LLM_API_KEY", None),
        help="LLM API key (default: $LLM_API_KEY)")
    parser.add_argument("--base-url",
        default=os.environ.get("LLM_BASE_URL", None),
        help="LLM base URL / Azure endpoint (default: $LLM_BASE_URL)")
    parser.add_argument("--api-version", "-a",
        default=os.environ.get("LLM_API_VERSION", None),
        help="Azure API version (default: $LLM_API_VERSION)")
    parser.add_argument("--reasoning-effort",
        default=os.environ.get("LLM_REASONING_EFFORT", None),
        choices=["low", "medium", "high"],
        help="Reasoning effort for o-series models (default: $LLM_REASONING_EFFORT)")
    parser.add_argument("--max-new-tokens",
        type=int,
        default=int(os.environ.get("LLM_MAX_NEW_TOKENS", "16384")),
        help="Max output tokens (default: $LLM_MAX_NEW_TOKENS or 16384)")
    parser.add_argument("--max-input-tokens",
        type=int,
        default=int(os.environ.get("LLM_MAX_INPUT_TOKENS", "120000")),
        help="Max input tokens for context truncation (default: $LLM_MAX_INPUT_TOKENS or 120000)")
    parser.add_argument("--temperature",
        type=float,
        default=float(os.environ.get("LLM_TEMPERATURE", "1.0")),
        help="Sampling temperature (default: $LLM_TEMPERATURE or 1.0)")
    parser.add_argument("--top-p",
        type=float,
        default=float(os.environ.get("LLM_TOP_P", "1.0")),
        help="Nucleus sampling probability (default: $LLM_TOP_P or 1.0)")
    parser.add_argument("--language",
        default=os.environ.get("LANGUAGE", "python"),
        choices=["python", "r"],
        help="Executor language: python|r (default: $LANGUAGE or python)")
    parser.add_argument("--specialty-prompt",
        default=os.environ.get("SPECIALTY_PROMPT", None),
        help="Domain-specific instructions appended to the system prompt (default: $SPECIALTY_PROMPT)")
    parser.add_argument("--db-connection-code",
        default=os.environ.get("DB_CONNECTION_CODE", None),
        help="Python/R code to establish a DB connection (default: $DB_CONNECTION_CODE)")
    parser.add_argument("--db-connection-file",
        default=os.environ.get("DB_CONNECTION_FILE", None),
        help="Path to a file containing DB connection code (default: $DB_CONNECTION_FILE)")
    parser.add_argument("--no-resume",
        action="store_true",
        help="Do not load Python state from a previous session")
    parser.add_argument("--query", "-q",
        default=None,
        help="Run a single query and exit (non-interactive)")
    parser.add_argument("--quiet",
        action="store_true",
        help="Suppress agent verbose output")

    args = parser.parse_args()
    console = Console()

    # Resolve DB connection code
    db_connection_code = args.db_connection_code
    if args.db_connection_file:
        try:
            with open(args.db_connection_file, "r") as f:
                db_connection_code = f.read()
        except Exception as e:
            console.print(f"[red]Failed to read DB connection file: {e}[/red]")
            sys.exit(1)

    # Build agent
    try:
        agent, python_tool, r_tool = create_agent(
            work_dir=args.work_dir,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            api_version=args.api_version,
            reasoning_effort=args.reasoning_effort,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            language=args.language,
            specialty_prompt=args.specialty_prompt,
            db_connection_code=db_connection_code,
            verbose=not args.quiet,
            resume_state=not args.no_resume,
        )
    except Exception as e:
        console.print(f"[red]Failed to create agent: {e}[/red]")
        sys.exit(1)

    work_dir = os.path.abspath(args.work_dir)

    if args.query:
        agent.run(args.query)
        if python_tool:
            save_state(python_tool, work_dir)
    else:
        _print_welcome(console, work_dir, args.model, args.language)
        interactive_mode(agent, console, work_dir, python_tool, r_tool)


if __name__ == "__main__":
    main()