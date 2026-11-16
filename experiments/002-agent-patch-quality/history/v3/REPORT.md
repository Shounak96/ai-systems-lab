# Experiment 002 V3 — five-scenario AutoSaddler V2 comparison

## Comparison

| Metric | Qwen3 1.7B | Qwen3 4B |
|---|---:|---:|
| Fixed-rubric diagnosis accuracy | 1/5 | 5/5 |
| Native-schema-valid responses | 5/5 | 5/5 |
| Verifier-valid/finalized mutations | 3/5 | 5/5 |
| Accepted training improvements | 1/5 | 4/5 |
| Verifier rejections | 2 | 0 |
| Evaluated training failures | 2 | 1 |
| Development-gated candidates | 1 | 4 |
| Development improvements / unchanged / regressions | 1 / 0 / 0 | 2 / 1 / 1 |
| Prompt tokens | 2,325 | 2,295 |
| Generated tokens | 1,017 | 965 |
| Ollama inference latency | 82.629 s | 180.152 s |

Both models used Q4_K_M quantization, temperature 0, seed 42, `num_ctx=4096`, `num_predict=1200`, thinking disabled, and zero retries. The ten calls ran sequentially on CPU through Ollama 0.40.16.

## Training and development outcomes

| Scenario | Qwen3 1.7B | Qwen3 4B |
|---|---|---|
| Structured output | Mutation evaluated; train 0→0; development not gated in | Accepted; train 0→1; development 0.5→0.0; regressed `dev-result-dict-copy` |
| Tool arguments | Verifier rejected a forbidden `for` loop; train-after unevaluated | Accepted; train 0→1; development 0.5→0.5; no measured regression |
| Retries | Accepted; train 0→1; development 0→0.5 | Accepted; train 0→1; development 0→0.5 |
| State management | Verifier rejected a forbidden `for` loop; train-after unevaluated | Accepted; train 0→1; development 0→1.0 |
| Error handling | Mutation evaluated; train 0→0; development not gated in | Mutation evaluated; train 0→0; development not gated in |

For 1.7B, one of three evaluated mutations improved training; two additional mutations were rejected before evaluation. Its only accepted mutation improved development by 0.5. For 4B, four of five mutations improved training and were accepted. Their development changes were −0.5, 0, +0.5, and +1.0: one full development pass, two partial outcomes, and one regression.

## Verification

- Preregistration was frozen before model execution at `sha256:f0c0603b824ae815bc93cf0c5a92a79985d48eb74ed09ee9aaf526398e11f877`.
- The preflight suite passed all 19 tests, including five known-good mutations through the official evidence, mutation, matched-training, acceptance, and development-gate workflow.
- Raw-artifact audit verified ten result rows, exactly one diagnosis terminal event per model/scenario, matching Ollama token counters, the preregistration digest, and byte-identical normalized model-facing requests for all five scenarios.
- The official AutoSaddler checkout remained clean at `e841543ce6cf6bdf690e65134b12a829ec9f66d1`. Earlier pilot and V2 result artifacts were not overwritten.

## Limitations

- Five synthetic, handcrafted scenarios and one deterministic call per model are an exploratory comparison, not a general benchmark or reproduction of AutoSaddler's paper results.
- Diagnosis accuracy uses frozen substring concept groups. It is reproducible but brittle: some semantically relevant 1.7B explanations did not contain a preregistered term and remained scored incorrect; no post-hoc rescoring was performed.
- Strict training acceptance does not guarantee development generalization. In particular, one 4B mutation passed training but reduced development score and introduced a regression.
- The narrow preregistered AST verifier rejected two 1.7B updates containing loops. These are contract-adherence failures, not evaluated behavioral failures.
- Isolation uses disposable directories, a stripped environment, socket blocking, timeouts, and an AST allowlist; it is not a hardened container or separate OS security boundary.
- Repeated post-run test-suite executions intermittently encountered Windows `PermissionError` failures in official `LocalRunStore` atomic projection replacement. The full suite passed before inference; V3 tests and raw audit passed after inference, and a failed deterministic test passed immediately in isolation. The engine was not modified and failures were not hidden with retry logic.

Further evaluation is justified because the 4B model showed materially higher diagnosis, verifier-validity, and training-acceptance rates. It should remain limited and preregistered, with more independent scenarios/seeds and stronger OS-level isolation; the mixed development results do not yet support a reliability claim.
