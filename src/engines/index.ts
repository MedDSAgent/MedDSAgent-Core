import OpenAI, { AzureOpenAI } from "openai";
import type {
  ChatCompletionMessageParam,
  ChatCompletionTool,
} from "openai/resources/chat/completions";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export type { ChatCompletionMessageParam as Message, ChatCompletionTool as ToolSchema };

export interface ToolCallResult {
  name: string;
  arguments: Record<string, unknown>;
}

export interface ChatResult {
  response: string;
  toolCalls: ToolCallResult[];
}

// ---------------------------------------------------------------------------
// LLMEngine interface
// ---------------------------------------------------------------------------

export interface LLMEngine {
  chat(messages: ChatCompletionMessageParam[], tools: ChatCompletionTool[]): Promise<ChatResult>;
}

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

interface BaseConfig {
  model: string;
  maxTokens?: number;
  temperature?: number;
  /** Extra params forwarded verbatim to the API call */
  extra?: Record<string, unknown>;
}

/**
 * Config for any OpenAI-compatible endpoint.
 * Covers: OpenAI, vLLM, SGLang, OpenRouter, and any other provider that
 * implements the OpenAI chat completions API.
 * Set `baseURL` to point at a non-OpenAI host (e.g. http://localhost:8000/v1).
 * Providers that don't require an API key (e.g. local vLLM) can set `apiKey`
 * to any non-empty string — the openai SDK requires a value but the server
 * will ignore it.
 */
export interface OpenAIConfig extends BaseConfig {
  provider: "openai" | "vllm" | "sglang" | "openrouter" | (string & {});
  apiKey: string;
  baseURL?: string;
}

export interface AzureOpenAIConfig extends BaseConfig {
  provider: "azure";
  apiKey: string;
  endpoint: string;
  apiVersion?: string;
  /** Azure deployment name — defaults to `model` */
  deployment?: string;
}

export type EngineConfig = OpenAIConfig | AzureOpenAIConfig;

// ---------------------------------------------------------------------------
// Argument normalization
// LLMs sometimes return `arguments` as a pre-serialized JSON string instead of
// a parsed object. Normalize to Record<string, unknown> in both cases.
// ---------------------------------------------------------------------------

function normalizeArguments(args: string | Record<string, unknown> | null | undefined): Record<string, unknown> {
  if (!args) return {};
  if (typeof args === "string") {
    try {
      return JSON.parse(args) as Record<string, unknown>;
    } catch {
      return { raw: args };
    }
  }
  return args;
}

// ---------------------------------------------------------------------------
// Shared response parsing
// ---------------------------------------------------------------------------

function parseResponse(message: OpenAI.Chat.Completions.ChatCompletionMessage): ChatResult {
  const response = message.content ?? "";
  const toolCalls: ToolCallResult[] = (message.tool_calls ?? []).map((tc) => ({
    name: tc.function.name,
    arguments: normalizeArguments(tc.function.arguments),
  }));
  return { response, toolCalls };
}

// ---------------------------------------------------------------------------
// OpenAIEngine
// ---------------------------------------------------------------------------

export class OpenAIEngine implements LLMEngine {
  private readonly client: OpenAI;
  private readonly model: string;
  private readonly callParams: Record<string, unknown>;

  constructor(config: OpenAIConfig) {
    this.client = new OpenAI({
      apiKey: config.apiKey,
      ...(config.baseURL ? { baseURL: config.baseURL } : {}),
    });
    this.model = config.model;
    this.callParams = {
      ...(config.maxTokens != null ? { max_tokens: config.maxTokens } : {}),
      ...(config.temperature != null ? { temperature: config.temperature } : {}),
      ...(config.extra ?? {}),
    };
  }

  async chat(
    messages: ChatCompletionMessageParam[],
    tools: ChatCompletionTool[],
  ): Promise<ChatResult> {
    const response = await this.client.chat.completions.create({
      model: this.model,
      messages,
      stream: false,
      ...(tools.length > 0 ? { tools, tool_choice: "auto" as const } : {}),
      ...this.callParams,
    });

    const choice = response.choices[0];
    if (!choice) throw new Error("OpenAI returned empty choices.");
    return parseResponse(choice.message);
  }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/**
 * Instantiate the right engine from a config object.
 * Azure → AzureOpenAIEngine; everything else → OpenAIEngine.
 */
export function createEngine(config: EngineConfig): LLMEngine {
  if (config.provider === "azure") {
    return new AzureOpenAIEngine(config as AzureOpenAIConfig);
  }
  return new OpenAIEngine(config as OpenAIConfig);
}

// ---------------------------------------------------------------------------
// AzureOpenAIEngine
// ---------------------------------------------------------------------------

export class AzureOpenAIEngine implements LLMEngine {
  private readonly client: AzureOpenAI;
  private readonly deployment: string;
  private readonly callParams: Record<string, unknown>;

  constructor(config: AzureOpenAIConfig) {
    this.client = new AzureOpenAI({
      apiKey: config.apiKey,
      endpoint: config.endpoint,
      apiVersion: config.apiVersion ?? "2024-10-21",
      deployment: config.deployment ?? config.model,
    });
    this.deployment = config.deployment ?? config.model;
    this.callParams = {
      ...(config.maxTokens != null ? { max_tokens: config.maxTokens } : {}),
      ...(config.temperature != null ? { temperature: config.temperature } : {}),
      ...(config.extra ?? {}),
    };
  }

  async chat(
    messages: ChatCompletionMessageParam[],
    tools: ChatCompletionTool[],
  ): Promise<ChatResult> {
    const response = await this.client.chat.completions.create({
      model: this.deployment,
      messages,
      stream: false,
      ...(tools.length > 0 ? { tools, tool_choice: "auto" as const } : {}),
      ...this.callParams,
    });

    const choice = response.choices[0];
    if (!choice) throw new Error("AzureOpenAI returned empty choices.");
    return parseResponse(choice.message);
  }
}
