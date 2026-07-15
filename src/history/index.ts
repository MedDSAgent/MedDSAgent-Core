import { randomUUID } from "crypto";

// ---------------------------------------------------------------------------
// Payload types used inside steps
// ---------------------------------------------------------------------------

export interface ToolCall {
  tool_name: string;
  tool_args: string; // JSON string
  tool_title: string;
}

export interface ToolOutput {
  tool_name: string;
  output: string;
}

// ---------------------------------------------------------------------------
// Step union
// ---------------------------------------------------------------------------

export type StepType = "SystemStep" | "UserStep" | "AgentStep" | "ObservationStep";

interface StepBase {
  stepId: string;
  startTime: string; // ISO-8601
  endTime: string;
}

export interface SystemStep extends StepBase {
  type: "SystemStep";
  systemMessage: string;
}

export interface UserStep extends StepBase {
  type: "UserStep";
  userInput: string;
}

export interface AgentStep extends StepBase {
  type: "AgentStep";
  agentId: string;
  response: string;
  tools: ToolCall[];
  isFinal: boolean;
}

export interface ObservationStep extends StepBase {
  type: "ObservationStep";
  agentId: string;
  toolOutputs: ToolOutput[];
}

export type Step = SystemStep | UserStep | AgentStep | ObservationStep;

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

function now(): string {
  return new Date().toISOString();
}

export function makeSystemStep(systemMessage: string, opts?: { stepId?: string; startTime?: string; endTime?: string }): SystemStep {
  const t = opts?.startTime ?? now();
  return {
    type: "SystemStep",
    stepId: opts?.stepId ?? randomUUID(),
    startTime: t,
    endTime: opts?.endTime ?? t,
    systemMessage,
  };
}

export function makeUserStep(userInput: string, opts?: { stepId?: string; startTime?: string; endTime?: string }): UserStep {
  const t = opts?.startTime ?? now();
  return {
    type: "UserStep",
    stepId: opts?.stepId ?? randomUUID(),
    startTime: t,
    endTime: opts?.endTime ?? t,
    userInput,
  };
}

export function makeAgentStep(
  agentId: string,
  opts?: {
    response?: string;
    tools?: ToolCall[];
    isFinal?: boolean;
    stepId?: string;
    startTime?: string;
    endTime?: string;
  },
): AgentStep {
  const t = opts?.startTime ?? now();
  return {
    type: "AgentStep",
    stepId: opts?.stepId ?? randomUUID(),
    startTime: t,
    endTime: opts?.endTime ?? t,
    agentId,
    response: opts?.response ?? "",
    tools: opts?.tools ?? [],
    isFinal: opts?.isFinal ?? false,
  };
}

export function makeObservationStep(
  agentId: string,
  toolOutputs: ToolOutput[],
  opts?: { stepId?: string; startTime?: string; endTime?: string },
): ObservationStep {
  const t = opts?.startTime ?? now();
  return {
    type: "ObservationStep",
    stepId: opts?.stepId ?? randomUUID(),
    startTime: t,
    endTime: opts?.endTime ?? t,
    agentId,
    toolOutputs,
  };
}

// ---------------------------------------------------------------------------
// Serialization (keep keys snake_case to match Python DB blobs exactly)
// ---------------------------------------------------------------------------

type RawStep = Record<string, unknown>;

export function serializeStep(step: Step): RawStep {
  switch (step.type) {
    case "SystemStep":
      return {
        type: "SystemStep",
        step_id: step.stepId,
        start_time: step.startTime,
        end_time: step.endTime,
        system_message: step.systemMessage,
      };
    case "UserStep":
      return {
        type: "UserStep",
        step_id: step.stepId,
        start_time: step.startTime,
        end_time: step.endTime,
        user_input: step.userInput,
      };
    case "AgentStep":
      return {
        type: "AgentStep",
        step_id: step.stepId,
        agent_id: step.agentId,
        start_time: step.startTime,
        end_time: step.endTime,
        response: step.response,
        tools: step.tools,
        is_final: step.isFinal,
      };
    case "ObservationStep":
      return {
        type: "ObservationStep",
        step_id: step.stepId,
        agent_id: step.agentId,
        start_time: step.startTime,
        end_time: step.endTime,
        tool_outputs: step.toolOutputs,
      };
  }
}

export function deserializeStep(raw: RawStep): Step {
  const type = raw["type"] as StepType;
  const stepId = (raw["step_id"] as string | undefined) ?? randomUUID();
  const startTime = raw["start_time"] as string;
  const endTime = raw["end_time"] as string;

  switch (type) {
    case "SystemStep":
      return { type, stepId, startTime, endTime, systemMessage: raw["system_message"] as string };
    case "UserStep":
      return { type, stepId, startTime, endTime, userInput: raw["user_input"] as string };
    case "AgentStep":
      return {
        type,
        stepId,
        startTime,
        endTime,
        agentId: raw["agent_id"] as string,
        response: (raw["response"] as string | undefined) ?? "",
        tools: (raw["tools"] as ToolCall[] | undefined) ?? [],
        isFinal: (raw["is_final"] as boolean | undefined) ?? false,
      };
    case "ObservationStep":
      return {
        type,
        stepId,
        startTime,
        endTime,
        agentId: raw["agent_id"] as string,
        toolOutputs: (raw["tool_outputs"] as ToolOutput[] | undefined) ?? [],
      };
    default:
      throw new Error(`Unknown step type: ${String(type)}`);
  }
}

// ---------------------------------------------------------------------------
// Round
// ---------------------------------------------------------------------------

export interface RoundData {
  roundNum: number;
  steps: Step[];
}

export function makeRound(roundNum: number): RoundData {
  return { roundNum, steps: [] };
}

export function getUserStep(round: RoundData): UserStep {
  const s = round.steps.find((s): s is UserStep => s.type === "UserStep");
  if (!s) throw new Error("No UserStep found in round.");
  return s;
}

export function getAnswerStep(round: RoundData): AgentStep {
  const s = round.steps.find((s): s is AgentStep => s.type === "AgentStep" && s.isFinal);
  if (!s) throw new Error("No final AgentStep found in round.");
  return s;
}

export function serializeRound(round: RoundData): RawStep {
  return {
    round_num: round.roundNum,
    steps: round.steps.map(serializeStep),
  };
}

export function deserializeRound(raw: RawStep): RoundData {
  const roundNum = raw["round_num"] as number;
  const steps = ((raw["steps"] as RawStep[]) ?? []).map(deserializeStep);
  return { roundNum, steps };
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

export interface HistoryData {
  rounds: RoundData[];
  currentRoundNum: number;
}

export function makeHistory(): HistoryData {
  return { rounds: [], currentRoundNum: 0 };
}

export function addStep(history: HistoryData, step: Step): void {
  if (step.type === "UserStep") {
    history.currentRoundNum += 1;
    const newRound = makeRound(history.currentRoundNum);
    newRound.steps.push(step);
    history.rounds.push(newRound);
    return;
  }

  if (history.rounds.length === 0) {
    if (step.type === "SystemStep") {
      const newRound = makeRound(0);
      newRound.steps.push(step);
      history.rounds.push(newRound);
      return;
    }
    throw new Error("Cannot add non-UserStep before any UserStep.");
  }

  const currentRound = history.rounds[history.rounds.length - 1];
  if (!currentRound) throw new Error("No current round.");
  currentRound.steps.push(step);
}

export function serializeHistory(history: HistoryData): RawStep {
  return { rounds: history.rounds.map(serializeRound) };
}

export function deserializeHistory(raw: RawStep): HistoryData {
  const rounds = ((raw["rounds"] as RawStep[]) ?? []).map(deserializeRound);
  const currentRoundNum = rounds.reduce((max, r) => Math.max(max, r.roundNum), 0);
  return { rounds, currentRoundNum };
}
