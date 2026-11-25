"""
CI regression gate: a small, fixed, seeded subset of the Monte Carlo
benchmark (scripts/benchmark.py --ci).  The false-support count must not rise
above the stored baseline (benchmarks/ci_baseline.json), i.e. a change may
never make an *incorrect* location easier to report as ``supported``.
Regenerate the baseline deliberately with
``python scripts/benchmark.py --ci --write-ci-baseline`` and explain why.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import benchmark  # noqa: E402


def test_benchmark_subset_false_support_not_worse_than_baseline():
    with open(benchmark.CI_BASELINE) as fh:
        base = json.load(fh)
    out = benchmark.run_benchmark(benchmark.CI_RUNS, benchmark.CI_SEED, benchmark.CI_CONFIG, workers=1, quiet=True)
    s = out["summary"]["overall"]
    assert s["runs"] == base["runs"]
    assert s["false_supported"] <= base["false_supported"], (
        f"false supports rose from {base['false_supported']} to {s['false_supported']} "
        f"(seeds {s['false_support_seeds']})")
    assert s["false_support_rate"] <= base["false_support_rate"] + 1e-12
    # Guard against a vacuous pass: the pipeline must still produce located results.
    assert s["with_location"] >= max(1, base["with_location"] - 1)
