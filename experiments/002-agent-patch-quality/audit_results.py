from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def main() -> int:
    run = json.loads((ARTIFACTS / "results.json").read_text(encoding="utf-8"))
    prompt_hashes: dict[str, dict[str, str]] = defaultdict(dict)
    raw_metric_identity = True

    for row in run["results"]:
        raw_path = ARTIFACTS / "raw" / f"{row['model'].replace(':', '_')}-{row['scenario']}.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        response = raw["response"]
        raw_metric_identity &= (
            response.get("model") == row["model"]
            and response.get("eval_count") == row["generated_tokens"]
            and response.get("prompt_eval_count") == row["prompt_tokens"]
        )
        prompt_hashes[row["scenario"]][row["model"]] = hashlib.sha256(
            raw["prompt"].encode("utf-8")
        ).hexdigest()

    identical_prompts = all(len(set(items.values())) == 1 for items in prompt_hashes.values())
    if len(run["results"]) != 10 or not raw_metric_identity or not identical_prompts:
        raise SystemExit("raw-results audit failed")

    summaries = {}
    for model in ("qwen3:1.7b", "qwen3:4b"):
        rows = [row for row in run["results"] if row["model"] == model]
        total_latency = sum(row["latency_seconds"] or 0 for row in rows)
        summaries[model] = {
            "root_cause_accurate": sum(bool(row["root_cause_accurate"]) for row in rows),
            "patch_applicable": sum(bool(row["patch_applicable"]) for row in rows),
            "failing_test_resolved": sum(bool(row["failing_test_resolved"]) for row in rows),
            "regression_free": sum(bool(row["regression_free"]) for row in rows),
            "prompt_tokens": sum(row["prompt_tokens"] or 0 for row in rows),
            "generated_tokens": sum(row["generated_tokens"] or 0 for row in rows),
            "latency_seconds_total": round(total_latency, 6),
            "latency_seconds_mean": round(total_latency / len(rows), 6),
        }

    audit = {
        "raw_metric_identity": raw_metric_identity,
        "identical_prompt_hashes_between_models": identical_prompts,
        "result_rows": len(run["results"]),
        "summaries": summaries,
    }
    (ARTIFACTS / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
