"""
Generate reasoning traces from base GPT-OSS for harmony-channel SFT retraining.

For each example in sft_train.jsonl / sft_val.jsonl, calls the base gpt-oss model
and captures both the reasoning (analysis channel → reasoning_content) and the
final answer (final channel → content).

Output: data/finetune/art{N}/traces_{train,val}.jsonl
Format per record:
  {
    "messages": [{"role": "user", "content": "..."}],
    "reasoning_content": "...",   # analysis channel — may be empty if server omits it
    "answer": "{...}"             # GOLD label JSON from original sft_*.jsonl (not model output)
  }

Usage (standalone — polls for the endpoint file to appear):
  python echr/finetune/gen_reasoning_traces.py \
      --articles art3 art6 art8 \
      --endpoint_file data/vllm_gptoss_base_endpoint.txt \
      --model gpt-oss-120b-base \
      --max_workers 8
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


def call_gptoss(client: OpenAI, user_content: str, model: str, max_tokens: int, idx: int):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=max_tokens,
                temperature=0.0,
                seed=42,
            )
            msg = resp.choices[0].message
            content = msg.content or ""
            # reasoning_content is gpt-oss specific (vLLM extension for analysis channel)
            reasoning = getattr(msg, "reasoning_content", None) or ""
            finish = resp.choices[0].finish_reason
            return idx, reasoning, content, finish
        except Exception as e:
            if attempt < 2:
                print(f"  [idx={idx}] retry {attempt+1}/3: {e}", flush=True)
                time.sleep(5)
            else:
                print(f"  [idx={idx}] FAILED: {e}", flush=True)
                return idx, "", "", "error"


def generate_traces(article_dir: Path, client: OpenAI, model: str,
                    max_workers: int, max_tokens: int):
    for split in ("train", "val"):
        src = article_dir / f"sft_{split}.jsonl"
        dst = article_dir / f"traces_{split}.jsonl"

        if not src.exists():
            print(f"  {src} not found, skipping", flush=True)
            continue

        if dst.exists():
            existing = sum(1 for _ in dst.open())
            total = sum(1 for _ in src.open())
            if existing == total:
                print(f"  {dst} already complete ({existing} records), skipping", flush=True)
                continue
            print(f"  {dst} partial ({existing}/{total}), regenerating", flush=True)

        examples = [json.loads(l) for l in src.open() if l.strip()]
        print(f"  {split}: {len(examples)} examples → {dst}", flush=True)

        results = [None] * len(examples)
        n_done = 0

        def task(i_ex):
            i, ex = i_ex
            msgs = ex["messages"]
            user_content = next(m["content"] for m in msgs if m["role"] == "user")
            gold_answer = next(
                (m["content"] for m in msgs if m["role"] == "assistant"), ""
            )
            idx, reasoning, content, finish = call_gptoss(
                client, user_content, model, max_tokens, i
            )
            return idx, reasoning, gold_answer, finish

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(task, (i, ex)) for i, ex in enumerate(examples)]
            for fut in as_completed(futs):
                i, reasoning, gold_answer, finish = fut.result()
                results[i] = {
                    "messages": [{"role": "user", "content": examples[i]["messages"][0]["content"]}],
                    "reasoning_content": reasoning,
                    "answer": gold_answer,
                }
                n_done += 1
                if n_done % 50 == 0 or n_done == len(examples):
                    n_null = sum(1 for r in results if r is not None and not r["reasoning_content"])
                    print(f"    {n_done}/{len(examples)} done, {n_null} null reasoning", flush=True)

        with dst.open("w") as f:
            for rec in results:
                f.write(json.dumps(rec) + "\n")

        n_null = sum(1 for r in results if not r["reasoning_content"])
        print(f"  Saved {len(results)} records to {dst} ({n_null} null reasoning)", flush=True)


def wait_for_server(endpoint: str, timeout_s: int = 1800):
    client = OpenAI(api_key="EMPTY", base_url=endpoint)
    print(f"Waiting for server at {endpoint}...", flush=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            client.models.list()
            print("Server ready!", flush=True)
            return client
        except Exception:
            time.sleep(15)
    raise RuntimeError(f"Server {endpoint} did not become ready within {timeout_s}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", nargs="+", default=["art3", "art6", "art8"])
    parser.add_argument("--data_dir", default="data/finetune")
    parser.add_argument("--endpoint_file", default="data/vllm_gptoss_base_endpoint.txt")
    parser.add_argument("--model", default="gpt-oss-120b-base")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=2000)
    args = parser.parse_args()

    ep_file = Path(args.endpoint_file)
    print(f"Waiting for endpoint file: {ep_file}", flush=True)
    while not ep_file.exists():
        time.sleep(15)
    endpoint = ep_file.read_text().strip()
    print(f"Endpoint: {endpoint}", flush=True)

    client = wait_for_server(endpoint)

    for article in args.articles:
        art_dir = Path(args.data_dir) / article
        print(f"\n=== {article} ({art_dir}) ===", flush=True)
        generate_traces(art_dir, client, args.model, args.max_workers, args.max_tokens)

    print("\nAll articles done.", flush=True)


if __name__ == "__main__":
    main()
