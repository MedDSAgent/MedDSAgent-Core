# Run the agent in CLI

Build first (`npm run build`), then:

```bash
node dist/cli/index.js chat \
    --work-dir /home/daviden1013/David_projects/temp/sessions/backend_test \
    --provider openai \
    --base-url https://openrouter.ai/api/v1 \
    --api-key "$OPENROUTER_API_KEY" \
    --model openai/gpt-oss-120b \
    --reasoning-effort low \
    --temperature 1.0
```

OpenRouter (and vLLM, SGLang, and any other OpenAI-compatible endpoint) is reached
through `--provider openai` with `--base-url` pointed at it; there is no separate
provider flag for them. `--provider azure` is the only other value.

Start the HTTP server instead with:

```bash
node dist/cli/index.js serve --work-dir ./workspace --port 7842
```
