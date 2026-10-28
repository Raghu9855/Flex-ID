"""FLEX-ID Performance and Load Testing Harness.

Benchmarks latency distributions (p50, p90, p95, p99), throughput (RPS),
and error rates across critical API endpoints under concurrent traffic.

Usage:
    # Benchmark in-process via ASGI transport (default, no server setup needed)
    python scripts/load_test.py

    # Benchmark external/live server (e.g., local uvicorn or remote deployment)
    python scripts/load_test.py --url http://127.0.0.1:8000 --concurrency 20 --requests 200
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List

import httpx
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app  # noqa: E402


# Representative synthetic network flow payload (29 numerical CIC-IDS2018 features)
SAMPLE_FLOW: Dict[str, Any] = {
    "features": [
        80.0, 6.0, 150000.0, 25.0, 30.0, 4500.0, 12000.0,
        500.0, 120.0, 800.0, 210.0, 450.0, 110.0, 150.0,
        25000.0, 350.0, 1200.0, 45.0, 50000.0, 20.0, 10.0,
        2.0, 1.0, 0.0, 1.0, 0.0, 512.0, 1024.0, 15.0
    ],
    "model_version": "fedprox-v1"
}

BATCH_PAYLOAD: Dict[str, Any] = {
    "records": [{"features": SAMPLE_FLOW["features"]} for _ in range(10)],
    "model_version": "fedprox-v1"
}

EXPLANATION_PAYLOAD: Dict[str, Any] = {
    "features": SAMPLE_FLOW["features"],
    "model_version": "fedprox-v1",
    "top_k": 5
}


async def send_request(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    json_data: Any = None
) -> float:
    """Send a single request and return the latency in milliseconds.

    Raises Exception on HTTP status >= 400.
    """
    start = time.perf_counter()
    if method == "GET":
        resp = await client.get(path)
    elif method == "POST":
        resp = await client.post(path, json=json_data)
    else:
        raise ValueError(f"Unsupported method: {method}")

    latency_ms = (time.perf_counter() - start) * 1000.0
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:100]}")
    return latency_ms


async def run_scenario(
    client: httpx.AsyncClient,
    name: str,
    method: str,
    path: str,
    json_data: Any,
    total_requests: int,
    concurrency: int
) -> Dict[str, Any]:
    """Execute a concurrent load test scenario."""
    print(f"\n[Running] {name}: {total_requests} requests (concurrency={concurrency})...")
    sem = asyncio.Semaphore(concurrency)
    latencies: List[float] = []
    errors: int = 0

    async def worker():
        nonlocal errors
        async with sem:
            try:
                lat = await send_request(client, method, path, json_data)
                latencies.append(lat)
            except Exception:
                errors += 1

    start_wall = time.perf_counter()
    tasks = [asyncio.create_task(worker()) for _ in range(total_requests)]
    await asyncio.gather(*tasks)
    total_wall_time = time.perf_counter() - start_wall

    successful = len(latencies)
    rps = successful / total_wall_time if total_wall_time > 0 else 0.0

    if latencies:
        p50 = float(np.percentile(latencies, 50))
        p90 = float(np.percentile(latencies, 90))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        avg = float(np.mean(latencies))
        min_lat = float(np.min(latencies))
        max_lat = float(np.max(latencies))
    else:
        p50 = p90 = p95 = p99 = avg = min_lat = max_lat = 0.0

    result = {
        "scenario": name,
        "endpoint": f"{method} {path}",
        "total_requests": total_requests,
        "concurrency": concurrency,
        "successful": successful,
        "failed": errors,
        "wall_time_sec": round(total_wall_time, 3),
        "rps": round(rps, 1),
        "latency_ms": {
            "min": round(min_lat, 2),
            "avg": round(avg, 2),
            "p50": round(p50, 2),
            "p90": round(p90, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "max": round(max_lat, 2)
        }
    }
    return result


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """Print results in a clean markdown/ASCII table."""
    print("\n" + "=" * 88)
    print(f"{'Endpoint / Scenario':<28} | {'Reqs':<5} | {'RPS':<7} | {'p50(ms)':<8} | {'p95(ms)':<8} | {'p99(ms)':<8} | {'Errors':<6}")
    print("-" * 88)
    for r in results:
        l = r["latency_ms"]
        print(f"{r['scenario']:<28} | {r['successful']:<5} | {r['rps']:<7.1f} | {l['p50']:<8.2f} | {l['p95']:<8.2f} | {l['p99']:<8.2f} | {r['failed']:<6}")
    print("=" * 88)


async def main():
    parser = argparse.ArgumentParser(description="FLEX-ID API Performance Benchmark")
    parser.add_argument("--url", type=str, default="", help="Base URL of live server (empty for in-process ASGI)")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent client workers (default: 10)")
    parser.add_argument("--requests", type=int, default=100, help="Requests per scenario (default: 100)")
    parser.add_argument("--output", type=str, default="results/load_test_results.json", help="Output JSON path")
    args = parser.parse_args()

    scenarios = [
        ("Health Check", "GET", "/health", None, args.requests, args.concurrency),
        ("List Models", "GET", "/api/v1/models", None, args.requests, args.concurrency),
        ("Single Prediction", "POST", "/api/v1/predictions", SAMPLE_FLOW, args.requests, args.concurrency),
        ("Batch Prediction (x10)", "POST", "/api/v1/predictions/batch", BATCH_PAYLOAD, max(20, args.requests // 2), args.concurrency),
        ("SHAP Explanation", "POST", "/api/v1/explanations", EXPLANATION_PAYLOAD, max(15, args.requests // 4), min(5, args.concurrency)),
    ]

    results = []

    if args.url:
        print(f"Benchmarking against external server at {args.url}")
        async with httpx.AsyncClient(base_url=args.url, timeout=30.0) as client:
            for name, method, path, data, reqs, conc in scenarios:
                res = await run_scenario(client, name, method, path, data, reqs, conc)
                results.append(res)
    else:
        print("Benchmarking in-process via ASGI transport (lifespan active)...")
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=30.0) as client:
            for name, method, path, data, reqs, conc in scenarios:
                res = await run_scenario(client, name, method, path, data, reqs, conc)
                results.append(res)

    print_summary_table(results)

    # Save to JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed metrics saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
