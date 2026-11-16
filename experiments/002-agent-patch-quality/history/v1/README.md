# Experiment 002: local agent patch quality

This project is a five-scenario exploratory pilot asking whether two differently sized, locally hosted Qwen3 models can diagnose small agent-harness failures and produce regression-free patches. It is inspired by AutoSaddler's diagnosis-and-patch framing, but it neither reproduces that framework nor supports comparisons with its benchmark results.

The scenarios cover structured-output validation, tool argument handling, retries, state management, and error handling. Each model receives the same source, task, failure trace, output contract, seed, and generation settings. Thinking is disabled so the fixed 1,400-token generation budget is used for the requested patch response. Hidden tests are only loaded after patch generation.

## Environment

- Python 3.12.10 in project-local `.venv`
- Standard library only; no packages installed
- Preferred local models: `qwen3:1.7b` and `qwen3:4b`
- Ollama endpoint: `http://127.0.0.1:11434`

## Commands

From this directory, using the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe pilot.py self-test
.\.venv\Scripts\python.exe pilot.py run
.\.venv\Scripts\python.exe audit_results.py
```

The run command never pulls models. It fails closed and records a blocker if Ollama or both preferred model sizes are unavailable. Candidate patches are limited to `harness.py`, checked before application, statically rejected for obvious I/O/process primitives, and evaluated in isolated temporary copies with credentials removed from the subprocess environment and sockets disabled by the test runner.

Results are written to `artifacts/results.json`, `artifacts/results.csv`, and `REPORT.md`. Successful inference runs also retain exact prompts, raw model responses, Ollama token counters, model metadata, and test output under `artifacts/raw/`.

## Interpretation

Five handcrafted synthetic cases are enough to validate the experiment mechanics and expose obvious model differences, but they are not a general benchmark. Finite hidden tests and keyword-based diagnosis scoring can miss behaviors. Report only observed outcomes and do not infer broad model quality.
