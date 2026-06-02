#!/usr/bin/env python3
"""
Benchmark AI chatbot/search/recommendation service.

Measures:
- Chatbot latency: avg, p50, p95, min, max, error rate.
- Search accuracy: precision@k, recall@k, hit@k, MRR when expected_product_ids are provided.
- Recommendation quality: precision@k, recall@k, hit@k when relevant_product_ids are provided.
- Throughput: requests/second under concurrent load.

Example:
    python3 benchmark_ai_service.py --base-url http://localhost:5001/api/ai
    python3 benchmark_ai_service.py --cases benchmark_cases.example.json --output benchmark_report.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CASES: dict[str, list[dict[str, Any]]] = {
    "search": [
        {"name": "shoes-running", "query": "giay chay bo nhe thoang khi", "top_k": 5},
        {"name": "gym-clothes", "query": "ao quan tap gym co gian thoang mat", "top_k": 5},
        {"name": "color-size", "query": "ao mau den size M con hang", "top_k": 5},
    ],
    "recommendation": [
        {"name": "user-1", "user_id": 1, "limit": 6},
    ],
    "chat": [
        {"name": "fit-question", "message": "Toi cao 170cm nang 68kg nen mac size nao?", "user_id": 1},
        {"name": "preference-question", "message": "Toi thuong thich san pham nao?", "user_id": 1},
        {"name": "attribute-question", "message": "San pham nao co mau den va size M?", "user_id": 1},
    ],
}


@dataclass
class TimedResponse:
    ok: bool
    status_code: int | None
    latency_ms: float
    data: dict[str, Any] | None
    error: str | None = None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_latencies(results: list[TimedResponse]) -> dict[str, Any]:
    latencies = [r.latency_ms for r in results if r.ok]
    errors = [r for r in results if not r.ok]
    return {
        "count": len(results),
        "success": len(latencies),
        "errors": len(errors),
        "error_rate": round(len(errors) / len(results), 4) if results else 0.0,
        "avg_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "min_ms": round(min(latencies), 2) if latencies else None,
        "p50_ms": round(percentile(latencies, 0.50), 2) if latencies else None,
        "p95_ms": round(percentile(latencies, 0.95), 2) if latencies else None,
        "max_ms": round(max(latencies), 2) if latencies else None,
    }


def timed_post(url: str, payload: dict[str, Any], timeout: float) -> TimedResponse:
    started = time.perf_counter()
    encoded = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status_code = response.status
            data = json.loads(body.decode("utf-8")) if body else {}
        latency_ms = (time.perf_counter() - started) * 1000
        return TimedResponse(
            ok=200 <= status_code < 300,
            status_code=status_code,
            latency_ms=latency_ms,
            data=data,
            error=None,
        )
    except urllib.error.HTTPError as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        body = exc.read()
        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            data = {"raw": body.decode("utf-8", errors="replace")[:500]}
        return TimedResponse(False, exc.code, latency_ms, data, str(data)[:500])
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return TimedResponse(False, None, latency_ms, None, str(exc))


def extract_ids(data: dict[str, Any] | None) -> list[int]:
    if not data:
        return []

    candidates = [
        data.get("product_ids"),
        data.get("suggested_product_ids"),
        data.get("data"),
    ]

    for value in candidates:
        if isinstance(value, list):
            ids: list[int] = []
            for item in value:
                if isinstance(item, dict) and "id" in item:
                    ids.append(int(item["id"]))
                elif isinstance(item, (int, float, str)):
                    try:
                        ids.append(int(item))
                    except ValueError:
                        pass
            if ids:
                return ids

    nested_data = data.get("data")
    if isinstance(nested_data, dict):
        return extract_ids(nested_data)
    return []


def ranking_metrics(predicted: list[int], relevant: list[int], k: int) -> dict[str, float | None]:
    if not relevant:
        return {
            "precision_at_k": None,
            "recall_at_k": None,
            "hit_at_k": None,
            "mrr": None,
        }

    top = predicted[:k]
    relevant_set = set(relevant)
    hits = [pid for pid in top if pid in relevant_set]

    reciprocal_rank = 0.0
    for rank, pid in enumerate(top, start=1):
        if pid in relevant_set:
            reciprocal_rank = 1.0 / rank
            break

    return {
        "precision_at_k": round(len(hits) / max(len(top), 1), 4),
        "recall_at_k": round(len(hits) / len(relevant_set), 4),
        "hit_at_k": 1.0 if hits else 0.0,
        "mrr": round(reciprocal_rank, 4),
    }


def mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return round(statistics.mean(values), 4) if values else None


def load_cases(path: str | None) -> dict[str, list[dict[str, Any]]]:
    if not path:
        return DEFAULT_CASES
    with open(path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return {
        "search": loaded.get("search", []),
        "recommendation": loaded.get("recommendation", []),
        "chat": loaded.get("chat", []),
    }


def benchmark_search(base_url: str, cases: list[dict[str, Any]], timeout: float) -> dict[str, Any]:
    details = []
    responses: list[TimedResponse] = []
    for case in cases:
        payload = {
            "query": case["query"],
            "top_k": int(case.get("top_k", 5)),
        }
        if case.get("category_id") is not None:
            payload["category_id"] = case["category_id"]
        response = timed_post(f"{base_url}/search", payload, timeout)
        responses.append(response)
        predicted = extract_ids(response.data)
        relevant = [int(x) for x in case.get("expected_product_ids", [])]
        metrics = ranking_metrics(predicted, relevant, int(case.get("top_k", 5)))
        details.append({
            "name": case.get("name", case["query"]),
            "query": case["query"],
            "predicted_product_ids": predicted,
            "expected_product_ids": relevant,
            "latency_ms": round(response.latency_ms, 2),
            "ok": response.ok,
            "error": response.error,
            **metrics,
        })

    return {
        "latency": summarize_latencies(responses),
        "quality": {
            "precision_at_k": mean_metric(details, "precision_at_k"),
            "recall_at_k": mean_metric(details, "recall_at_k"),
            "hit_at_k": mean_metric(details, "hit_at_k"),
            "mrr": mean_metric(details, "mrr"),
            "evaluated_cases": sum(1 for row in details if row["precision_at_k"] is not None),
        },
        "details": details,
    }


def benchmark_recommendation(base_url: str, cases: list[dict[str, Any]], timeout: float) -> dict[str, Any]:
    details = []
    responses: list[TimedResponse] = []
    for case in cases:
        payload = {
            "user_id": int(case["user_id"]),
            "limit": int(case.get("limit", 6)),
        }
        response = timed_post(f"{base_url}/recommend", payload, timeout)
        responses.append(response)
        predicted = extract_ids(response.data)
        relevant = [int(x) for x in case.get("relevant_product_ids", [])]
        metrics = ranking_metrics(predicted, relevant, int(case.get("limit", 6)))
        details.append({
            "name": case.get("name", f"user-{case['user_id']}"),
            "user_id": int(case["user_id"]),
            "predicted_product_ids": predicted,
            "relevant_product_ids": relevant,
            "latency_ms": round(response.latency_ms, 2),
            "ok": response.ok,
            "error": response.error,
            **metrics,
        })

    return {
        "latency": summarize_latencies(responses),
        "quality": {
            "precision_at_k": mean_metric(details, "precision_at_k"),
            "recall_at_k": mean_metric(details, "recall_at_k"),
            "hit_at_k": mean_metric(details, "hit_at_k"),
            "mrr": mean_metric(details, "mrr"),
            "evaluated_cases": sum(1 for row in details if row["precision_at_k"] is not None),
        },
        "details": details,
    }


def benchmark_chat(base_url: str, cases: list[dict[str, Any]], timeout: float) -> dict[str, Any]:
    details = []
    responses: list[TimedResponse] = []
    for index, case in enumerate(cases):
        payload = {
            "message": case.get("message") or case.get("query"),
            "query": case.get("query") or case.get("message"),
        }
        if case.get("user_id") is not None:
            payload["user_id"] = int(case["user_id"])
        if case.get("session_id"):
            payload["session_id"] = case["session_id"]

        response = timed_post(f"{base_url}/chat", payload, timeout)
        responses.append(response)
        reply = ""
        if response.data:
            reply = str(response.data.get("reply") or response.data.get("data", {}).get("reply") or "")
        details.append({
            "name": case.get("name", f"chat-{index + 1}"),
            "message": payload["message"],
            "latency_ms": round(response.latency_ms, 2),
            "ok": response.ok,
            "reply_chars": len(reply),
            "product_ids": extract_ids(response.data),
            "error": response.error,
        })

    return {
        "latency": summarize_latencies(responses),
        "details": details,
    }


def throughput_payload(kind: str, cases: dict[str, list[dict[str, Any]]]) -> tuple[str, dict[str, Any]]:
    if kind == "search":
        case = (cases.get("search") or DEFAULT_CASES["search"])[0]
        return "/search", {"query": case["query"], "top_k": int(case.get("top_k", 5))}
    if kind == "recommend":
        case = (cases.get("recommendation") or DEFAULT_CASES["recommendation"])[0]
        return "/recommend", {"user_id": int(case["user_id"]), "limit": int(case.get("limit", 6))}
    case = (cases.get("chat") or DEFAULT_CASES["chat"])[0]
    return "/chat", {
        "message": case.get("message") or case.get("query"),
        "query": case.get("query") or case.get("message"),
        "user_id": case.get("user_id"),
    }


def benchmark_throughput(
    base_url: str,
    cases: dict[str, list[dict[str, Any]]],
    kind: str,
    total_requests: int,
    concurrency: int,
    timeout: float,
) -> dict[str, Any]:
    endpoint, payload = throughput_payload(kind, cases)
    url = f"{base_url}{endpoint}"
    started = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(timed_post, url, payload, timeout) for _ in range(total_requests)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    duration_s = time.perf_counter() - started
    successes = sum(1 for result in results if result.ok)
    return {
        "endpoint": endpoint,
        "kind": kind,
        "total_requests": total_requests,
        "concurrency": concurrency,
        "duration_s": round(duration_s, 3),
        "requests_per_second": round(successes / duration_s, 3) if duration_s > 0 else None,
        "success": successes,
        "errors": total_requests - successes,
        "latency": summarize_latencies(results),
    }


def write_csv(report: dict[str, Any], output_path: str) -> str:
    csv_path = str(Path(output_path).with_suffix(".csv"))
    rows = []
    for section in ("search", "recommendation", "chat"):
        for detail in report.get(section, {}).get("details", []):
            rows.append({"section": section, **detail})

    if not rows:
        return csv_path

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AI service quality and performance.")
    parser.add_argument("--base-url", default="http://localhost:5001/api/ai", help="AI service base URL.")
    parser.add_argument("--cases", help="JSON file containing search/recommendation/chat cases.")
    parser.add_argument("--output", default="benchmark_report.json", help="JSON report output path.")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout per request in seconds.")
    parser.add_argument("--throughput-kind", choices=["search", "recommend", "chat"], default="search")
    parser.add_argument("--throughput-requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--skip-throughput", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    cases = load_cases(args.cases)

    report: dict[str, Any] = {
        "base_url": base_url,
        "generated_at_unix": int(time.time()),
        "search": benchmark_search(base_url, cases.get("search", []), args.timeout),
        "recommendation": benchmark_recommendation(base_url, cases.get("recommendation", []), args.timeout),
        "chat": benchmark_chat(base_url, cases.get("chat", []), args.timeout),
    }

    if not args.skip_throughput:
        report["throughput"] = benchmark_throughput(
            base_url=base_url,
            cases=cases,
            kind=args.throughput_kind,
            total_requests=args.throughput_requests,
            concurrency=args.concurrency,
            timeout=args.timeout,
        )

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    csv_path = write_csv(report, args.output)

    print(json.dumps({
        "output": args.output,
        "csv": csv_path,
        "search_quality": report["search"]["quality"],
        "recommendation_quality": report["recommendation"]["quality"],
        "chat_latency": report["chat"]["latency"],
        "throughput": report.get("throughput"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
