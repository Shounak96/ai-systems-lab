# Experiment 002 pilot report

This five-scenario handcrafted study is an exploratory pilot, not a general benchmark or an AutoSaddler reproduction.

## Setup

- Ollama 0.40.16, CPU execution
- `qwen3:1.7b`: reported 2.0B parameters, Q4_K_M, digest `8f68893c685c`
- `qwen3:4b`: 4.0B parameters, Q4_K_M, digest `359d7dd4bcda`
- Intel Core 5 120U, 12 logical CPUs, 15.72 GiB RAM, Intel integrated graphics
- Identical prompts and settings: temperature 0, seed 42, thinking disabled, 1,400 generated-token cap

## Scenario results

`RC/P/A/F` means root cause passed the deterministic keyword rubric / patch applicable / exposed failure resolved / full hidden suite regression-free.

| Scenario | Qwen3 1.7B | Qwen3 4B |
|---|---|---|
| Structured output | 0/0/0/0; 110 tokens; 7.98 s | 0/0/0/0; 1,400 tokens; 189.26 s |
| Tool arguments | 1/0/0/0; 116 tokens; 7.95 s | 0/0/0/0; 1,400 tokens; 174.42 s |
| Retries | 0/0/0/0; 212 tokens; 12.91 s | 0/0/0/0; 1,400 tokens; 185.21 s |
| State management | 0/0/0/0; 156 tokens; 9.86 s | 0/0/0/0; 1,400 tokens; 183.28 s |
| Error handling | 1/1/0/0; 285 tokens; 17.12 s | 0/0/0/0; 1,400 tokens; 187.05 s |

## Totals

| Model | Root cause | Applicable | Exposed fixed | Regression-free | Prompt tokens | Generated tokens | Total / mean latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 1.7B | 2/5 | 1/5 | 0/5 | 0/5 | 1,257 | 879 | 55.82 / 11.16 s |
| Qwen3 4B | 0/5 | 0/5 | 0/5 | 0/5 | 1,227 | 7,000 | 919.22 / 183.84 s |

## Failures and interpretation

The 1.7B model returned schema-valid JSON on all cases, but two responses supplied replacement code instead of unified diffs, two diffs were corrupt, and its only applicable diff failed both exposed and hidden tests. Its raw retry and state diagnoses were substantively relevant despite failing the deliberately strict keyword rubric.

The 4B model emitted extended reasoning as ordinary response text even with thinking disabled. Every case exhausted the 1,400-token limit, ended with `done_reason=length`, and was truncated before a schema-valid JSON patch response. It was therefore much slower and produced no evaluable patches. This measures behavior under this fixed prompt and cap, not general model quality.

The raw-response audit confirmed ten rows, matching model identities and token counters, and identical prompt hashes between models. Six automated harness tests passed after the run. Finite synthetic cases, keyword diagnosis scoring, CPU-only execution, and one deterministic seed materially limit inference.

## Recommendation

Do not expand to a larger dataset yet. A follow-up pilot is justified only after a pre-registered output strategy can produce schema-valid, applicable patches—especially controlling the 4B model's verbose reasoning—without post-hoc repair. The current configuration produced zero exposed-test or regression-free fixes from either model.
