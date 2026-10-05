# model-router

Routes each request to the right LLM. **Jev** (TypeSafe's System One model, called via
OpenRouter) decides; your code owns the constraints and the policy.

## How a request flows

1. **Constraints in code.** Context length and vision needs filter the registry first.
   If only one model is eligible, Jev isn't even called.
2. **Jev decides.** It gets the request as `state` and one `choice` question whose options
   are the eligible models' descriptions. It returns a pick with probabilities and confidence.
3. **Policy in a threshold.** Confidence below `confidence_threshold` escalates to the
   strongest eligible model. If Jev is down, the registry `default` is used.
4. **Execute with fallback.** Chosen model first, then stronger, then weaker, using each model's configured provider.
5. **Log** every decision to `logs/decisions.jsonl`.

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    cp .env.example .env   # add OPENROUTER_API_KEY, then: export $(cat .env | xargs)
    ollama pull qwen2.5:7b # required for the configured local Ollama model
    pytest

## Use

    python -m model_router --dry-run "Summarize the tradeoffs of Kafka vs SQS"
    python -m model_router "Write a haiku about routers"
    python evals/run_eval.py

## Edit these first

- `config/models.yaml`: your models, descriptions, costs, threshold, and provider. OpenRouter IDs are sent through OpenRouter; models with `provider: ollama` are sent to the local Ollama server.
- `evals/eval_set.jsonl`: grow to ~100 labeled prompts from real traffic.

## Notes

- The configured `ollama-local` model uses `qwen2.5:7b` at `OLLAMA_BASE_URL` (default
  `http://localhost:11434`). Change its `id` to a model tag installed in Ollama if needed.
- The configured DeepSeek model uses OpenRouter (`deepseek/deepseek-chat`) and requires
  `OPENROUTER_API_KEY`; replace its ID with an available OpenRouter model ID if needed.
- Jev on OpenRouter was in beta; the endpoint is configurable via `JEV_ENDPOINT`
  (`/api/alpha/decisions` by default; the newer `/api/v1/systemone` takes the same body shape
  with bare model ids like `jev-1.13`). Check the OpenRouter docs if a call fails.
- The prompt text is sent to Jev (and truncated to 4000 chars). Set `preview_chars=0` in
  `DecisionLog` if you don't want prompt text in logs.
- Extending Jev's role: add more questions to the same call (e.g. a `noul` for "needs tools")
  and combine them in `Router.route`. One request, many judgments.
