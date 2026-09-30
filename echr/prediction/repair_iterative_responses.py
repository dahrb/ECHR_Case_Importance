"""Repair iterative records that contain a missing level response.

The main iterative runner now fails before writing these records.  This helper
repairs artifacts produced before that guard existed, replacing only affected
records atomically after all four level calls succeed.
"""

import argparse
import json
import os
from pathlib import Path

from echr.prediction.run_iterative_predictions import (
    LEVELS,
    get_client,
    load_prompt_records,
    predict_iterative,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"])
    parser.add_argument("--level-concurrency", type=int, default=4)
    args = parser.parse_args()

    output_path = Path(args.output)
    records = [json.loads(line) for line in output_path.open()]
    prompts = {record["Filename"]: record for record in load_prompt_records(args.prompt_dir)}
    affected = [
        record for record in records
        if any(not value.get("raw") for value in record.get("level_results", {}).values())
    ]
    if not affected:
        print("No missing iterative responses to repair.")
        return

    client = get_client(args.endpoint)
    replacements = {}
    for index, record in enumerate(affected, 1):
        filename = record["Filename"]
        prompt_record = prompts[filename]
        repaired = predict_iterative(
            client, prompt_record, args.article, args.model, int(prompt_record["text"]),
            level_prompts=prompt_record["level_prompts"], max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort, level_concurrency=args.level_concurrency,
        )
        if any(not value.get("raw") for value in repaired["level_results"].values()):
            raise RuntimeError(f"Repair failed for {filename}")
        replacements[filename] = {
            "Filename": filename,
            "importance": record.get("importance"),
            **repaired,
        }
        print(f"Repaired {index}/{len(affected)}: {filename}", flush=True)

    temporary = output_path.with_suffix(output_path.suffix + ".repairing")
    with temporary.open("w") as handle:
        for record in records:
            handle.write(json.dumps(replacements.get(record["Filename"], record)) + "\n")
    os.replace(temporary, output_path)
    print(f"Repaired {len(replacements)} records in {output_path}")


if __name__ == "__main__":
    main()
