"""Simple asynchronous load generator for OpenAI-compatible servers (e.g. vLLM)."""

import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from openai import AsyncOpenAI
from transformers import AutoTokenizer


@dataclass
class RequestResult:
    """
    Stores benchmark metrics from a single request.

    Attributes:
        prompt_len_target (int): Target prompt length requested from CLI.
        prompt_len (int): Actual number of tokens in the input prompt
            (including chat template).
        out_len (int): Number of tokens requested in the output (max_tokens).
        concurrency (int): Concurrency level used for the request.
        ttft (float): Time To First Token (in seconds).
        latency (float): Total request latency (in seconds).
    """

    prompt_len_target: int
    prompt_len: int
    out_len: int
    concurrency: int
    ttft: float
    latency: float


def build_chat_for_target_tokens(
    tokenizer,
    target_tokens: int,
    system_prompt: str,
    base_story: str,
) -> Tuple[List[Dict[str, str]], int]:
    """
    Build (system, user) messages from a long base_story such that the
    *formatted* prompt (with the chat template) is as close as possible
    to `target_tokens` tokens, without exceeding it.

    Returns:
        messages: list of {role, content} messages to send.
        actual_len: token length of tokenizer.apply_chat_template(..., tokenize=True).
    """
    full_messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': base_story},
    ]
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_len = len(full_ids)

    if full_len <= target_tokens:
        return full_messages, full_len

    low, high = 1, len(base_story)
    best_text = ''
    best_len = 0

    while low <= high:
        mid = (low + high) // 2
        candidate_text = base_story[:mid]
        candidate_messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': candidate_text},
        ]
        candidate_ids = tokenizer.apply_chat_template(
            candidate_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        L = len(candidate_ids)

        if L <= target_tokens:
            best_text = candidate_text
            best_len = L
            low = mid + 1
        else:
            high = mid - 1

    final_messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': best_text},
    ]
    return final_messages, best_len


async def run_one_request(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, str]],
    prompt_len_actual: int,
    prompt_len_target: int,
    out_len: int,
    concurrency: int,
) -> RequestResult:
    """
    Sends a single streaming request and measures performance metrics.

    Args:
        client (AsyncOpenAI): The OpenAI client instance.
        model (str): Model name to send to the API.
        messages (List[Dict[str, str]]): Precomputed chat messages.
        prompt_len_actual (int): Actual token length of the formatted prompt.
        prompt_len_target (int): Target prompt length requested from CLI.
        out_len (int): Number of output tokens (max_tokens).
        concurrency (int): Concurrency level being tested.

    Returns:
        RequestResult: Recorded TTFT and latency for the request.
    """
    start = time.perf_counter()
    first_token_time = None

    stream = await client.chat.completions.create(
        model=model,
        stream=True,
        max_tokens=out_len,
        temperature=0.0,
        messages=messages,
        # We deactivate thinking to have a comparable benchmark
        extra_body={
            'chat_template_kwargs': {'enable_thinking': False},
        },
    )

    async for chunk in stream:
        if first_token_time is None:
            first_token_time = time.perf_counter()

    end = time.perf_counter()

    if first_token_time is None:
        first_token_time = end

    return RequestResult(
        prompt_len_target=prompt_len_target,
        prompt_len=prompt_len_actual,
        out_len=out_len,
        concurrency=concurrency,
        ttft=first_token_time - start,
        latency=end - start,
    )


async def run_config(
    client: AsyncOpenAI,
    model: str,
    prompt_len_target: int,
    prompt_len_actual: int,
    messages: List[Dict[str, str]],
    out_len: int,
    concurrency: int,
    num_requests: int,
) -> List[RequestResult]:
    """Runs a batch of requests for a given configuration and concurrency level.

    Args:
        client (AsyncOpenAI): The OpenAI client instance.
        model (str): Model to benchmark against.
        prompt_len (int): Prompt size in tokens.
        out_len (int): Number of output tokens to request.
        concurrency (int): How many requests to run simultaneously.
        num_requests (int): How many total requests to launch.

    Returns:
        List[RequestResult]: Collected metrics for each request."""

    print(
        f'[bench] target_prompt_len={prompt_len_target}, '
        f'actual_prompt_len={prompt_len_actual}, out_len={out_len}, '
        f'concurrency={concurrency}, num_requests={num_requests}'
    )

    results: List[RequestResult] = []

    remaining = num_requests
    while remaining > 0:
        batch_size = min(remaining, concurrency)
        tasks = [
            asyncio.create_task(
                run_one_request(
                    client=client,
                    model=model,
                    messages=messages,
                    prompt_len_actual=prompt_len_actual,
                    prompt_len_target=prompt_len_target,
                    out_len=out_len,
                    concurrency=concurrency,
                )
            )
            for _ in range(batch_size)
        ]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        remaining -= batch_size

    return results


def aggregate(results: List[RequestResult]) -> List[Dict[str, Any]]:
    """
    Aggregates raw request results into summary statistics.

    Groups results by (prompt_len_target, concurrency).

    Notes:
        - prompt_len_target: what you asked for (256, 2048, 8192, ...)
        - prompt_len: actual formatted prompt length in tokens
          (system + user + chat template).
    """
    rows: List[Dict[str, Any]] = []

    key_to_results: Dict[tuple, List[RequestResult]] = {}
    for r in results:
        key = (r.prompt_len_target, r.concurrency)
        key_to_results.setdefault(key, []).append(r)

    for (prompt_len_target, concurrency), group in sorted(key_to_results.items()):
        ttfts = np.array([g.ttft for g in group], dtype=float)
        lats = np.array([g.latency for g in group], dtype=float)
        prompt_len_actual = group[0].prompt_len
        out_len = group[0].out_len
        total_tokens_per_req = prompt_len_actual + out_len
        total_tokens = total_tokens_per_req * len(group)
        total_time = float(lats.sum())
        toks_per_sec = total_tokens / total_time if total_time > 0 else 0.0

        row = {
            'prompt_len_target': prompt_len_target,
            'prompt_len_actual': prompt_len_actual,
            'out_len': out_len,
            'concurrency': concurrency,
            'num_requests': len(group),
            'ttft_p50_s': float(np.percentile(ttfts, 50)),
            'ttft_p95_s': float(np.percentile(ttfts, 95)),
            'lat_p50_s': float(np.percentile(lats, 50)),
            'lat_p95_s': float(np.percentile(lats, 95)),
            'tokens_per_second': toks_per_sec,
        }
        rows.append(row)

    return rows


async def main_async(args: argparse.Namespace) -> None:
    """Executes the full benchmark based on command-line args."""
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    with open(args.user_prompt_file, 'r', encoding='utf-8') as f:
        base_story = f.read()

    all_results: List[RequestResult] = []

    for prompt_len_target in args.prompt_lens:
        messages, prompt_len_actual = build_chat_for_target_tokens(
            tokenizer=tokenizer,
            target_tokens=prompt_len_target,
            system_prompt=args.system_prompt,
            base_story=base_story,
        )
        print(
            f'[bench] built prompt for target={prompt_len_target} '
            f'-> actual={prompt_len_actual} tokens'
        )

        for concurrency in args.concurrency:
            cfg_results = await run_config(
                client=client,
                model=args.model,
                prompt_len_target=prompt_len_target,
                prompt_len_actual=prompt_len_actual,
                messages=messages,
                out_len=args.out_len,
                concurrency=concurrency,
                num_requests=args.num_requests,
            )
            all_results.extend(cfg_results)

    rows = aggregate(all_results)

    fieldnames = [
        'prompt_len_target',
        'prompt_len_actual',
        'out_len',
        'concurrency',
        'num_requests',
        'ttft_p50_s',
        'ttft_p95_s',
        'lat_p50_s',
        'lat_p95_s',
        'tokens_per_second',
    ]

    df = pd.DataFrame(rows, columns=fieldnames)
    df.to_csv(args.output_csv, index=False)

    print(f'[bench] Wrote {len(rows)} rows to {args.output_csv}')


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the benchmark script."""
    parser = argparse.ArgumentParser(description='Simple vLLM load generator')
    parser.add_argument(
        '--base-url',
        default='http://localhost:8000/v1',
        help='OpenAI-compatible base URL (default: http://localhost:8000/v1)',
    )
    parser.add_argument(
        '--api-key',
        default='dummy',
        help='API key (vLLM ignores value but OpenAI client requires it)',
    )
    parser.add_argument(
        '--model',
        default='Qwen/Qwen3-0.6B',
        help='Model name to send in the OpenAI API requests',
    )
    parser.add_argument(
        '--prompt-lens',
        nargs='+',
        type=int,
        default=[256, 2048, 8192],
        help='Prompt lengths to test (in tokens, including chat template; target)',
    )
    parser.add_argument(
        '--out-len',
        type=int,
        default=128,
        help='Output length to request (max_tokens)',
    )
    parser.add_argument(
        '--concurrency',
        nargs='+',
        type=int,
        default=[1, 8, 32],
        help='Concurrency levels to test',
    )
    parser.add_argument(
        '--num-requests',
        type=int,
        default=32,
        help='Number of requests per (prompt_len, concurrency) pair',
    )
    parser.add_argument(
        '--output-csv',
        default='results.csv',
        help='Path to write aggregated metrics CSV',
    )
    parser.add_argument(
        '--system-prompt',
        default='Continue the given story.',
        help='System prompt used for all requests.',
    )
    parser.add_argument(
        '--user-prompt-file',
        type=str,
        default='./1_baseline/prompts/prompt.txt',
        help='Path to a .txt file containing a long base user prompt (story). '
        'If set, overrides --user-prompt.',
    )

    return parser.parse_args()


def main() -> None:
    """Main entrypoint when executed as a script."""
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
